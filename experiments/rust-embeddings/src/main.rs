use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config as XlmRobertaConfig, XLMRobertaModel};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokenizers::{tokenizer::TruncationParams, Tokenizer};

#[derive(Debug, Deserialize)]
struct TextItem {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct ProbeBatch {
    id: String,
    task: String,
    items: Vec<TextItem>,
}

#[derive(Debug, Deserialize)]
struct ProbeInput {
    model_id: String,
    revision: String,
    max_seq_length: usize,
    pad_token_id: u32,
    raw_inputs: Vec<TextItem>,
    batches: Vec<ProbeBatch>,
}

fn required_path(flag: &str, args: &[String]) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let value = args
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .ok_or_else(|| format!("missing required argument {flag}"))?;
    Ok(PathBuf::from(value))
}

fn pad_rows(
    encodings: &[tokenizers::Encoding],
    pad_token_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let width = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(0);
    let mut input_rows = Vec::with_capacity(encodings.len());
    let mut mask_rows = Vec::with_capacity(encodings.len());
    for encoding in encodings {
        let mut input_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();
        input_ids.resize(width, pad_token_id);
        attention_mask.resize(width, 0);
        input_rows.push(input_ids);
        mask_rows.push(attention_mask);
    }
    (input_rows, mask_rows)
}

fn infer_batch(
    batch: &ProbeBatch,
    tokenizer: &Tokenizer,
    model: &XLMRobertaModel,
    device: &Device,
    max_seq_length: usize,
    pad_token_id: u32,
) -> Result<(Value, BTreeMap<String, Vec<f32>>), Box<dyn Error + Send + Sync>> {
    if batch.items.is_empty() {
        return Err(format!("batch {} is empty", batch.id).into());
    }
    let prefix = match batch.task.as_str() {
        "query" => "query: ",
        "passage" => "passage: ",
        other => return Err(format!("batch {} has unknown task {other:?}", batch.id).into()),
    };
    let texts: Vec<String> = batch
        .items
        .iter()
        .map(|item| format!("{prefix}{}", item.text).trim().to_owned())
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer.encode_batch(text_refs, true)?;
    if encodings
        .iter()
        .any(|encoding| encoding.get_ids().len() > max_seq_length)
    {
        return Err(format!("batch {} was not truncated to max_seq_length", batch.id).into());
    }

    let (id_rows, mask_rows) = pad_rows(&encodings, pad_token_id);
    let batch_size = id_rows.len();
    let sequence_length = id_rows[0].len();
    let flat_ids: Vec<u32> = id_rows.iter().flatten().copied().collect();
    let flat_masks: Vec<u32> = mask_rows.iter().flatten().copied().collect();
    let flat_type_ids = vec![0_u32; batch_size * sequence_length];
    let input_ids = Tensor::from_vec(flat_ids, (batch_size, sequence_length), device)?;
    let attention_mask = Tensor::from_vec(flat_masks, (batch_size, sequence_length), device)?;
    let token_type_ids = Tensor::from_vec(flat_type_ids, (batch_size, sequence_length), device)?;

    let hidden = model.forward(
        &input_ids,
        &attention_mask,
        &token_type_ids,
        None,
        None,
        None,
    )?;
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
    let token_counts = mask.sum(1)?;
    let mean = summed.broadcast_div(&token_counts)?;
    let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
    let normalized = mean.broadcast_div(&norm)?;
    let vectors = normalized.to_vec2::<f32>()?;

    let mut captured = Map::new();
    let mut batch_vectors = BTreeMap::new();
    for (index, item) in batch.items.iter().enumerate() {
        captured.insert(
            item.id.clone(),
            json!({
                "input_ids": id_rows[index],
                "attention_mask": mask_rows[index],
            }),
        );
        batch_vectors.insert(item.id.clone(), vectors[index].clone());
    }
    Ok((Value::Object(captured), batch_vectors))
}

fn config_path(model_dir: &Path, name: &str) -> PathBuf {
    model_dir.join(name)
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = required_path("--model-dir", &args)?;
    let input_path = required_path("--inputs", &args)?;
    let output_path = required_path("--output", &args)?;
    let probe: ProbeInput = serde_json::from_slice(&fs::read(input_path)?)?;

    let model_config: XlmRobertaConfig =
        serde_json::from_slice(&fs::read(config_path(&model_dir, "config.json"))?)?;
    if model_config.pad_token_id != probe.pad_token_id {
        return Err(format!(
            "probe pad_token_id {} differs from model config {}",
            probe.pad_token_id, model_config.pad_token_id
        )
        .into());
    }
    if model_config.hidden_size != 768 || model_config.num_hidden_layers != 12 {
        return Err(format!(
            "unexpected XLM-R architecture: hidden_size={}, layers={}",
            model_config.hidden_size, model_config.num_hidden_layers
        )
        .into());
    }

    let device = Device::Cpu;
    let tokenizer_path = config_path(&model_dir, "tokenizer.json");
    let raw_tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    let mut model_tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    model_tokenizer.with_truncation(Some(TruncationParams {
        max_length: probe.max_seq_length,
        ..Default::default()
    }))?;
    if raw_tokenizer.token_to_id("<pad>") != Some(probe.pad_token_id) {
        return Err("tokenizer <pad> ID differs from pinned model config".into());
    }

    let weight_path = config_path(&model_dir, "model.safetensors");
    // Candle maps the pinned safetensors file and loads the requested XLM-R tensors on CPU.
    let weight_paths = [weight_path];
    let variables = unsafe {
        VarBuilder::from_mmaped_safetensors(&weight_paths, DType::F32, &device)?
    };
    let model = XLMRobertaModel::new(&model_config, variables)?;

    let mut raw_token_ids = Map::new();
    let mut raw_counts = Map::new();
    for item in &probe.raw_inputs {
        let encoding = raw_tokenizer.encode(item.text.as_str(), true)?;
        let ids = encoding.get_ids().to_vec();
        raw_counts.insert(item.id.clone(), json!(ids.len()));
        raw_token_ids.insert(item.id.clone(), json!(ids));
    }

    let mut batch_token_ids = Map::new();
    let mut batch_vectors_out = Map::new();
    let mut vectors = Map::new();
    for batch in &probe.batches {
        let (token_rows, vector_rows) = infer_batch(
            batch,
            &model_tokenizer,
            &model,
            &device,
            probe.max_seq_length,
            probe.pad_token_id,
        )?;
        batch_token_ids.insert(batch.id.clone(), token_rows);
        let mut vector_map = Map::new();
        for (id, vector) in vector_rows {
            vector_map.insert(id.clone(), json!(vector));
            vectors.entry(id).or_insert_with(|| json!(vector));
        }
        batch_vectors_out.insert(batch.id.clone(), Value::Object(vector_map));
    }

    let output = json!({
        "model_id": probe.model_id,
        "revision": probe.revision,
        "backend": "candle-xlm-roberta-cpu-f32",
        "max_seq_length": probe.max_seq_length,
        "architecture": {
            "model_type": "xlm-roberta",
            "hidden_size": model_config.hidden_size,
            "num_hidden_layers": model_config.num_hidden_layers,
            "num_attention_heads": model_config.num_attention_heads,
            "vocab_size": model_config.vocab_size,
            "pad_token_id": model_config.pad_token_id,
            "mean_pooling": true,
            "l2_normalization": true,
        },
        "raw_counts": raw_counts,
        "raw_token_ids": raw_token_ids,
        "batch_token_ids": batch_token_ids,
        "vectors": vectors,
        "batch_vectors": batch_vectors_out,
    });
    fs::write(output_path, serde_json::to_vec(&output)?)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rust-e5-probe: {error}");
        std::process::exit(1);
    }
}
