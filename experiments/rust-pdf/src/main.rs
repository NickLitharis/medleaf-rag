mod tables;

use std::{env, error::Error, path::Path};

use pdfium_render::prelude::*;
use serde::Serialize;

#[derive(Serialize)]
struct Probe {
    engine: &'static str,
    pages: Vec<Page>,
}

#[derive(Serialize)]
struct Page {
    page: usize,
    width_pt: f32,
    height_pt: f32,
    chars: Vec<Glyph>,
    tables: Vec<tables::Table>,
}

#[derive(Serialize)]
pub(crate) struct Glyph {
    pub(crate) index: usize,
    pub(crate) text: Option<String>,
    pub(crate) unicode_value: u32,
    pub(crate) generated: Option<bool>,
    pub(crate) font_weight: Option<u32>,
    /// Coordinates use PyMuPDF's top-left origin and PDF points.
    pub(crate) origin_pt: Option<[f32; 2]>,
    /// Coordinates use PyMuPDF's top-left origin: [left, top, right, bottom].
    pub(crate) bbox_pt: Option<[f32; 4]>,
}

fn font_weight_value(weight: PdfFontWeight) -> u32 {
    match weight {
        PdfFontWeight::Weight100 => 100,
        PdfFontWeight::Weight200 => 200,
        PdfFontWeight::Weight300 => 300,
        PdfFontWeight::Weight400Normal => 400,
        PdfFontWeight::Weight500 => 500,
        PdfFontWeight::Weight600 => 600,
        PdfFontWeight::Weight700Bold => 700,
        PdfFontWeight::Weight800 => 800,
        PdfFontWeight::Weight900 => 900,
        PdfFontWeight::Custom(value) => value,
    }
}

fn glyph_text(unicode_value: u32, is_hyphen: bool) -> Option<String> {
    // PDFium represents discretionary line-end hyphens as U+0002. Recover only
    // characters explicitly identified by FPDFText_IsHyphen; retain the raw code.
    let character = if unicode_value == 2 && is_hyphen {
        '-'
    } else {
        char::from_u32(unicode_value)?
    };
    Some(character.to_string())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let pdf_path = args.next().ok_or("usage: medleaf-ema-pdfium-probe <pdf-path>")?;
    if args.next().is_some() {
        return Err("usage: medleaf-ema-pdfium-probe <pdf-path>".into());
    }

    let library_path = env::var_os("PDFIUM_LIBRARY_PATH")
        .ok_or("PDFIUM_LIBRARY_PATH must point to the pinned libpdfium.so")?;
    let bindings = Pdfium::bind_to_library(Path::new(&library_path))?;
    let pdfium = Pdfium::new(bindings);
    let document = pdfium.load_pdf_from_file(Path::new(&pdf_path), None)?;

    let page_count = usize::try_from(document.pages().len())?;
    let mut pages = Vec::with_capacity(page_count);
    for (page_index, page) in document.pages().iter().enumerate() {
        let height = page.height().value;
        let text_page = page.text()?;
        let chars = text_page.chars();
        let mut glyphs = Vec::with_capacity(chars.len());

        for character in chars.iter() {
            let origin = character.origin().ok().map(|(x, y)| [x.value, height - y.value]);
            let bbox = character.loose_bounds().ok().map(|bounds| {
                [
                    bounds.left().value,
                    height - bounds.top().value,
                    bounds.right().value,
                    height - bounds.bottom().value,
                ]
            });

            let unicode_value = character.unicode_value();
            let is_hyphen = unicode_value == 2 && character.is_hyphen()?;
            glyphs.push(Glyph {
                index: character.index(),
                text: glyph_text(unicode_value, is_hyphen),
                unicode_value,
                generated: character.is_generated().ok(),
                font_weight: character.font_weight().map(font_weight_value),
                origin_pt: origin,
                bbox_pt: bbox,
            });
        }

        let tables = tables::extract_tables(&page, page_index + 1, height, &glyphs);

        pages.push(Page {
            page: page_index + 1,
            width_pt: page.width().value,
            height_pt: height,
            chars: glyphs,
            tables,
        });
    }

    let output = Probe {
        engine: "PDFium 158.0.8125.0 / pdfium-render 0.9.6",
        pages,
    };
    serde_json::to_writer(std::io::stdout().lock(), &output)?;
    println!();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("pdfium probe failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::glyph_text;

    #[test]
    fn recover_only_pdfium_identified_discretionary_hyphens() {
        assert_eq!(glyph_text(2, true).as_deref(), Some("-"));
        assert_eq!(glyph_text(2, false).as_deref(), Some("\u{2}"));
        assert_eq!(glyph_text('−' as u32, true).as_deref(), Some("−"));
        assert_eq!(glyph_text('0' as u32, true).as_deref(), Some("0"));
    }
}
