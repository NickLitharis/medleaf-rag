//! Detects ruled side-effect tables from vector rules and PDFium glyph geometry.
//! Ambiguous headers or frequency values are skipped with stderr diagnostics.

use std::collections::BTreeMap;

use pdfium_render::prelude::*;
use serde::Serialize;

use crate::Glyph;

// Coalesce sub-point coordinate drift across independently emitted PDF paths.
const COORD_TOLERANCE_PT: f32 = 1.25;
const INTERSECTION_TOLERANCE_PT: f32 = 1.75;
// Vector borders can be cut back around filled row dividers.
const GRAPH_JUNCTION_TOLERANCE_PT: f32 = 3.25;
const MIN_RULE_LENGTH_PT: f32 = 4.0;
const MAX_RULE_THICKNESS_PT: f32 = 2.75;
// Interior rules need repeated vertical support to define global columns.
const MIN_COLUMN_RULE_COVERAGE: f32 = 0.15;
const MIN_TEXT_COLUMN_GAP_PT: f32 = 5.0;
const TEXT_COLUMN_ALIGNMENT_TOLERANCE_PT: f32 = 5.0;

const FREQUENCY_LABELS: [&str; 14] = [
    "very common",
    "common",
    "uncommon",
    "rare",
    "very rare",
    "not known",
    "πολυ συχνες",
    "συχνες",
    "οχι συχνες",
    "σπανιες",
    "πολυ σπανιες",
    "μη γνωστης συχνοτητας",
    "μη γνωστες",
    "μη γνωστης",
];

#[derive(Serialize)]
pub(crate) struct Table {
    bbox_pt: [f32; 4],
    rows: Vec<TableRow>,
    lines: Vec<String>,
}

#[derive(Serialize)]
struct TableRow {
    cells: Vec<Option<TableCell>>,
}

#[derive(Serialize, Clone)]
struct TableCell {
    text: String,
    bbox_pt: [f32; 4],
    bold: Option<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    Horizontal,
    Vertical,
}

struct RawRule {
    axis: Axis,
    coordinate: f32,
    start: f32,
    end: f32,
}

struct RuleLine {
    axis: Axis,
    coordinate: f32,
    spans: Vec<(f32, f32)>,
}

struct Candidate {
    bbox: [f32; 4],
    xs: Vec<f32>,
    ys: Vec<f32>,
    horizontal: Vec<RuleLine>,
    vertical: Vec<RuleLine>,
}

struct DisjointSet {
    parents: Vec<usize>,
    ranks: Vec<u8>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parents: (0..count).collect(),
            ranks: vec![0; count],
        }
    }

    fn find(&mut self, item: usize) -> usize {
        let parent = self.parents[item];
        if parent != item {
            self.parents[item] = self.find(parent);
        }
        self.parents[item]
    }

    fn union(&mut self, first: usize, second: usize) {
        let mut first_root = self.find(first);
        let mut second_root = self.find(second);
        if first_root == second_root {
            return;
        }

        if self.ranks[first_root] < self.ranks[second_root] {
            std::mem::swap(&mut first_root, &mut second_root);
        }
        self.parents[second_root] = first_root;
        if self.ranks[first_root] == self.ranks[second_root] {
            self.ranks[first_root] += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct CellGlyph<'a> {
    glyph: &'a Glyph,
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    baseline: f32,
    height: f32,
    superscript: bool,
}

#[derive(Clone, Copy)]
enum TableKind {
    Grid,
    Labelled,
    Inline,
}

struct Header {
    row: usize,
    kind: TableKind,
    frequency_labels: Vec<Option<String>>,
}

pub(crate) fn extract_tables(
    page: &PdfPage<'_>,
    page_number: usize,
    page_height: f32,
    glyphs: &[Glyph],
) -> Vec<Table> {
    let rules = collect_rules(page, page_height);
    let candidates = find_candidates(rules);
    let mut tables = Vec::new();

    for mut candidate in candidates {
        infer_unruled_text_column(&mut candidate, glyphs);
        let grid = match reconstruct_grid(&candidate, glyphs) {
            Some(grid) => grid,
            None => {
                report_skip(page_number, &candidate, "could not reconstruct a rectangular cell grid");
                continue;
            }
        };
        if is_unstructured_single_cell_box(&grid) {
            report_skip(
                page_number,
                &candidate,
                "single-cell box has no explicit frequency-labelled value",
            );
            continue;
        }
        let lines = match find_header(&grid, &candidate.xs) {
            Ok(Some(header)) => flatten_rows(&grid, &header, page_number, &candidate),
            Ok(None) => {
                report_table_diagnostic(
                    page_number,
                    &candidate,
                    "header row has no text",
                );
                Vec::new()
            }
            Err(reason) => {
                report_table_diagnostic(page_number, &candidate, reason);
                Vec::new()
            }
        };

        tables.push(Table {
            bbox_pt: candidate.bbox,
            rows: grid
                .into_iter()
                .map(|cells| TableRow { cells })
                .collect(),
            lines,
        });
    }

    tables
}

fn collect_rules(page: &PdfPage<'_>, page_height: f32) -> Vec<RuleLine> {
    let mut raw = Vec::new();

    for object in page.objects().iter() {
        let PdfPageObject::Path(path) = object else {
            continue;
        };

        let stroked = path.is_stroked().unwrap_or(false)
            && path
                .stroke_width()
                .map(|width| width.value <= MAX_RULE_THICKNESS_PT)
                .unwrap_or(false);

        if stroked {
            if let Ok(matrix) = path.matrix() {
                let mut previous = None;
                for segment in path.segments().iter() {
                    let current = path_point_in_page(&matrix, segment.point(), page_height);

                    match segment.segment_type() {
                        PdfPathSegmentType::MoveTo => previous = Some(current),
                        PdfPathSegmentType::LineTo => {
                            if let Some((from_x, from_y)) = previous {
                                let delta_x = (current.0 - from_x).abs();
                                let delta_y = (current.1 - from_y).abs();
                                if delta_y <= 0.45 && delta_x >= MIN_RULE_LENGTH_PT {
                                    raw.push(RawRule {
                                        axis: Axis::Horizontal,
                                        coordinate: (current.1 + from_y) * 0.5,
                                        start: from_x.min(current.0),
                                        end: from_x.max(current.0),
                                    });
                                } else if delta_x <= 0.45 && delta_y >= MIN_RULE_LENGTH_PT {
                                    raw.push(RawRule {
                                        axis: Axis::Vertical,
                                        coordinate: (current.0 + from_x) * 0.5,
                                        start: from_y.min(current.1),
                                        end: from_y.max(current.1),
                                    });
                                }
                            }
                            previous = Some(current);
                        }
                        _ => previous = Some(current),
                    }
                }
            }
        }

        let filled = path
            .fill_mode()
            .map(|mode| mode != PdfPathFillMode::None)
            .unwrap_or(false);
        if filled {
            if let Ok(bounds) = path.bounds() {
                let rect = bounds.to_rect();
                let left = rect.left().value;
                let right = rect.right().value;
                let top = page_height - rect.top().value;
                let bottom = page_height - rect.bottom().value;
                let width = right - left;
                let height = bottom - top;

                if height > 0.0
                    && height <= MAX_RULE_THICKNESS_PT
                    && width >= MIN_RULE_LENGTH_PT
                {
                    raw.push(RawRule {
                        axis: Axis::Horizontal,
                        coordinate: (top + bottom) * 0.5,
                        start: left,
                        end: right,
                    });
                } else if width > 0.0
                    && width <= MAX_RULE_THICKNESS_PT
                    && height >= MIN_RULE_LENGTH_PT
                {
                    raw.push(RawRule {
                        axis: Axis::Vertical,
                        coordinate: (left + right) * 0.5,
                        start: top,
                        end: bottom,
                    });
                }
            }
        }
    }

    raw.sort_by(|first, second| {
        axis_order(first.axis)
            .cmp(&axis_order(second.axis))
            .then_with(|| first.coordinate.total_cmp(&second.coordinate))
            .then_with(|| first.start.total_cmp(&second.start))
    });

    let mut lines: Vec<RuleLine> = Vec::new();
    for rule in raw {
        let same_line = lines.last().is_some_and(|line| {
            line.axis == rule.axis && (line.coordinate - rule.coordinate).abs() <= COORD_TOLERANCE_PT
        });
        if same_line {
            let line = lines.last_mut().expect("last rule line exists");
            let count = line.spans.len() as f32;
            line.coordinate = (line.coordinate * count + rule.coordinate) / (count + 1.0);
            line.spans.push((rule.start, rule.end));
        } else {
            lines.push(RuleLine {
                axis: rule.axis,
                coordinate: rule.coordinate,
                spans: vec![(rule.start, rule.end)],
            });
        }
    }

    for line in &mut lines {
        line.spans
            .sort_by(|first, second| first.0.total_cmp(&second.0));
        let mut merged: Vec<(f32, f32)> = Vec::with_capacity(line.spans.len());
        for (start, end) in line.spans.drain(..) {
            if let Some(last) = merged.last_mut() {
                if start <= last.1 + INTERSECTION_TOLERANCE_PT {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            merged.push((start, end));
        }
        line.spans = merged;
    }

    lines
}
fn path_point_in_page(
    matrix: &PdfMatrix,
    point: (PdfPoints, PdfPoints),
    page_height: f32,
) -> (f32, f32) {
    let (x, pdf_y) = matrix.apply_to_points(point.0, point.1);
    (x.value, page_height - pdf_y.value)
}

fn axis_order(axis: Axis) -> u8 {
    match axis {
        Axis::Horizontal => 0,
        Axis::Vertical => 1,
    }
}

fn find_candidates(mut lines: Vec<RuleLine>) -> Vec<Candidate> {
    // Co-linear rules may belong to separate tables. Each disconnected span must
    // be its own graph vertex, rather than bridging the whitespace between them.
    let coalesced_count = lines.len();
    for index in 0..coalesced_count {
        while lines[index].spans.len() > 1 {
            let line = &mut lines[index];
            let segment = RuleLine {
                axis: line.axis,
                coordinate: line.coordinate,
                spans: vec![line.spans.pop().expect("multiple disconnected spans")],
            };
            lines.push(segment);
        }
    }
    if lines.len() != coalesced_count {
        lines.sort_unstable_by_key(|line| axis_order(line.axis));
    }
    let horizontal_count = lines
        .iter()
        .take_while(|line| line.axis == Axis::Horizontal)
        .count();
    let vertical_count = lines.len() - horizontal_count;
    if horizontal_count < 2 || vertical_count < 2 {
        return Vec::new();
    }

    let mut connected = DisjointSet::new(lines.len());
    for horizontal_index in 0..horizontal_count {
        let horizontal = &lines[horizontal_index];
        for vertical_index in horizontal_count..lines.len() {
            let vertical = &lines[vertical_index];
            if covers_point(horizontal, vertical.coordinate)
                && covers_point(vertical, horizontal.coordinate)
            {
                connected.union(horizontal_index, vertical_index);
            }
        }
    }

    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..lines.len() {
        let root = connected.find(index);
        components.entry(root).or_default().push(index);
    }

    let mut candidates = Vec::new();
    for indices in components.into_values() {
        let horizontal: Vec<RuleLine> = indices
            .iter()
            .filter_map(|&index| (lines[index].axis == Axis::Horizontal).then(|| clone_line(&lines[index])))
            .collect();
        let vertical: Vec<RuleLine> = indices
            .iter()
            .filter_map(|&index| (lines[index].axis == Axis::Vertical).then(|| clone_line(&lines[index])))
            .collect();
        if horizontal.len() < 2 || vertical.len() < 2 {
            continue;
        }

        let mut ys = unique_coords(horizontal.iter().map(|line| line.coordinate));
        if ys.len() < 2 {
            continue;
        }

        let mut xs = supported_column_edges(
            unique_coords(vertical.iter().map(|line| line.coordinate)),
            &vertical,
            ys[0],
            *ys.last().unwrap(),
        );
        if xs.len() < 2 {
            continue;
        }
        let first_horizontal = ys[0];
        let last_horizontal = *ys.last().unwrap();
        extend_unruled_table_bounds(&mut ys, &xs, &vertical);
        if ys[0] < first_horizontal || *ys.last().unwrap() > last_horizontal {
            xs = supported_column_edges(
                unique_coords(vertical.iter().map(|line| line.coordinate)),
                &vertical,
                ys[0],
                *ys.last().unwrap(),
            );
            if xs.len() < 2 {
                continue;
            }
        }

        let bbox = [xs[0], ys[0], *xs.last().unwrap(), *ys.last().unwrap()];
        if bbox[2] - bbox[0] < 18.0 || bbox[3] - bbox[1] < 14.0 {
            continue;
        }

        candidates.push(Candidate {
            bbox,
            xs,
            ys,
            horizontal,
            vertical,
        });
    }

    candidates.sort_by(|first, second| {
        first.bbox[1]
            .total_cmp(&second.bbox[1])
            .then_with(|| first.bbox[0].total_cmp(&second.bbox[0]))
    });
    candidates
}

// A continued table can omit a horizontal border at either page boundary.
fn extend_unruled_table_bounds(ys: &mut Vec<f32>, xs: &[f32], vertical: &[RuleLine]) {
    if ys.len() < 2 || xs.len() < 2 {
        return;
    }
    let first_horizontal = ys[0];
    let last_horizontal = *ys.last().unwrap();
    let left = xs[0];
    let right = *xs.last().unwrap();

    if let (Some((left_start, _)), Some((right_start, _))) = (
        vertical_edge_bounds(vertical, left, first_horizontal),
        vertical_edge_bounds(vertical, right, first_horizontal),
    ) {
        let top = left_start.min(right_start);
        if (left_start - right_start).abs() <= GRAPH_JUNCTION_TOLERANCE_PT
            && first_horizontal - top > GRAPH_JUNCTION_TOLERANCE_PT
        {
            ys.insert(0, top);
        }
    }
    if let (Some((_, left_end)), Some((_, right_end))) = (
        vertical_edge_bounds(vertical, left, last_horizontal),
        vertical_edge_bounds(vertical, right, last_horizontal),
    ) {
        let bottom = left_end.max(right_end);
        if (left_end - right_end).abs() <= GRAPH_JUNCTION_TOLERANCE_PT
            && bottom - last_horizontal > GRAPH_JUNCTION_TOLERANCE_PT
        {
            ys.push(bottom);
        }
    }
}

fn vertical_edge_bounds(
    rules: &[RuleLine],
    coordinate: f32,
    boundary: f32,
) -> Option<(f32, f32)> {
    let mut start = f32::INFINITY;
    let mut end = f32::NEG_INFINITY;
    for rule in rules {
        if (rule.coordinate - coordinate).abs() > COORD_TOLERANCE_PT {
            continue;
        }
        for &(span_start, span_end) in &rule.spans {
            if span_start <= boundary + GRAPH_JUNCTION_TOLERANCE_PT
                && span_end >= boundary - GRAPH_JUNCTION_TOLERANCE_PT
            {
                start = start.min(span_start);
                end = end.max(span_end);
            }
        }
    }
    start.is_finite().then_some((start, end))
}

fn clone_line(line: &RuleLine) -> RuleLine {
    RuleLine {
        axis: line.axis,
        coordinate: line.coordinate,
        spans: line.spans.clone(),
    }
}

fn covers_point(line: &RuleLine, point: f32) -> bool {
    line.spans.iter().any(|(start, end)| {
        point >= *start - GRAPH_JUNCTION_TOLERANCE_PT
            && point <= *end + GRAPH_JUNCTION_TOLERANCE_PT
    })
}

fn unique_coords(values: impl Iterator<Item = f32>) -> Vec<f32> {
    let mut values: Vec<f32> = values.collect();
    values.sort_by(f32::total_cmp);
    let mut result: Vec<f32> = Vec::new();
    let mut count = 0usize;
    for value in values {
        if let Some(last) = result.last_mut() {
            if (value - *last).abs() <= COORD_TOLERANCE_PT {
                count += 1;
                *last += (value - *last) / count as f32;
                continue;
            }
        }
        result.push(value);
        count = 1;
    }
    result
}

// Short one-row marks are cell content, not table-wide column boundaries.
fn supported_column_edges(
    mut coordinates: Vec<f32>,
    rules: &[RuleLine],
    start: f32,
    end: f32,
) -> Vec<f32> {
    let span = end - start;
    if span <= 0.0 {
        return Vec::new();
    }

    let mut intervals = Vec::new();
    let mut supported = 0;
    for index in 0..coordinates.len() {
        let coordinate = coordinates[index];
        intervals.clear();
        for rule in rules {
            if (rule.coordinate - coordinate).abs() > COORD_TOLERANCE_PT {
                continue;
            }
            for &(rule_start, rule_end) in &rule.spans {
                let clipped_start = rule_start.max(start);
                let clipped_end = rule_end.min(end);
                if clipped_end > clipped_start {
                    intervals.push((clipped_start, clipped_end));
                }
            }
        }
        intervals.sort_unstable_by(|first, second| first.0.total_cmp(&second.0));

        let mut covered = 0.0;
        if let Some(&(mut run_start, mut run_end)) = intervals.first() {
            for &(next_start, next_end) in &intervals[1..] {
                if next_start <= run_end {
                    run_end = run_end.max(next_end);
                } else {
                    covered += run_end - run_start;
                    run_start = next_start;
                    run_end = next_end;
                }
            }
            covered += run_end - run_start;
        }

        if covered / span >= MIN_COLUMN_RULE_COVERAGE {
            coordinates[supported] = coordinate;
            supported += 1;
        }
    }
    coordinates.truncate(supported);
    coordinates
}

// Some ruled tables use aligned text columns without an interior vector rule.
// Repeated explicit frequency labels provide native geometric evidence for one.
fn infer_unruled_text_column(candidate: &mut Candidate, glyphs: &[Glyph]) {
    let Some(divider) = infer_unruled_frequency_column(candidate, glyphs) else {
        return;
    };
    let heading_spans = infer_unruled_frequency_rows(candidate, glyphs, divider);
    let mut spans = Vec::new();
    let mut start = candidate.ys[0];
    for (top, bottom) in heading_spans {
        if top > start {
            spans.push((start, top));
        }
        start = bottom;
    }
    if start < *candidate.ys.last().unwrap() {
        spans.push((start, *candidate.ys.last().unwrap()));
    }
    candidate.xs.insert(1, divider);
    candidate.vertical.push(RuleLine {
        axis: Axis::Vertical,
        coordinate: divider,
        spans,
    });
}

fn infer_unruled_frequency_column(candidate: &Candidate, glyphs: &[Glyph]) -> Option<f32> {
    if candidate.xs.len() != 2 {
        return None;
    }

    let left = candidate.xs[0];
    let right = candidate.xs[1];
    let width = right - left;
    if width <= 0.0 {
        return None;
    }
    let min_anchor = left + width * 0.15;
    let max_anchor = right - width * 0.15;
    let top = candidate.ys[0];
    let bottom = *candidate.ys.last()?;
    let cell_glyphs = glyphs
        .iter()
        .filter_map(|glyph| {
            let text = glyph.text.as_deref()?;
            if text.trim().is_empty() {
                return None;
            }
            let (x, y) = glyph_position(glyph)?;
            if x < left - INTERSECTION_TOLERANCE_PT
                || x > right + INTERSECTION_TOLERANCE_PT
                || y < top - INTERSECTION_TOLERANCE_PT
                || y > bottom + INTERSECTION_TOLERANCE_PT
            {
                return None;
            }
            Some(CellGlyph::new(glyph))
        })
        .collect::<Vec<_>>();

    let mut lines = group_cell_lines(&cell_glyphs);
    for line in &mut lines {
        line.sort_by(|first, second| first.left.total_cmp(&second.left));
    }
    let mut anchors = Vec::new();
    for line in &lines {
        let Some(metrics) = line_metrics(line) else {
            continue;
        };
        let gap_threshold = (metrics.font_height * 0.8).max(MIN_TEXT_COLUMN_GAP_PT);
        let Some(first_text) = line.iter().position(|glyph| {
            glyph
                .glyph
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
        }) else {
            continue;
        };

        let mut previous: Option<usize> = None;
        let mut found = false;
        for index in first_text..line.len() {
            let glyph = &line[index];
            if glyph
                .glyph
                .text
                .as_deref()
                .map_or(true, |text| text.trim().is_empty())
            {
                continue;
            }
            if let Some(previous_index) = previous {
                let gap = glyph.left - line[previous_index].right;
                if gap >= gap_threshold && glyph.left >= min_anchor && glyph.left <= max_anchor {
                    let left_text = cell_text(&line[first_text..index]);
                    let right_text = cell_text(&line[index..]);
                    let left_label = line[first_text].left < min_anchor
                        && contains_frequency_label(&left_text);
                    if frequency_label(&right_text).is_some() || left_label {
                        anchors.push(glyph.left);
                        found = true;
                        break;
                    }
                }
            }
            previous = Some(index);
        }

        if !found {
            let first = &line[first_text];
            if first.left >= min_anchor
                && first.left <= max_anchor
                && frequency_label(&cell_text(&line[first_text..])).is_some()
            {
                anchors.push(first.left);
            }
        }
    }

    if anchors.len() < 2 {
        return None;
    }
    anchors.sort_by(f32::total_cmp);
    let anchor = anchors[anchors.len() / 2];
    let aligned = anchors
        .iter()
        .filter(|position| {
            (**position - anchor).abs() <= TEXT_COLUMN_ALIGNMENT_TOLERANCE_PT
        })
        .count();
    if aligned < 2 || aligned * 2 < anchors.len() {
        return None;
    }

    let divider = anchor - COORD_TOLERANCE_PT;
    if divider <= min_anchor || divider >= max_anchor {
        return None;
    }
    // Right-aligned inline frequency labels can resemble a text column.
    // A putative divider must not cut a complete label before its colon.
    for line in &lines {
        let Some(first) = line.first() else {
            continue;
        };
        let Some(colon) = line.iter().position(|glyph| {
            glyph.glyph.text.as_deref().is_some_and(|text| text.contains(':'))
        }) else {
            continue;
        };
        if first.left < divider && line[colon].right > divider {
            let text = cell_text(&line[..=colon]);
            if text.split_once(':').is_some_and(|(label, _)| frequency_label(label).is_some()) {
                return None;
            }
        }
    }
    Some(divider)
}

// A long unruled body can contain alternating full-width bold headings and
// frequency/reaction rows. Infer row boundaries only when both patterns recur.
fn infer_unruled_frequency_rows(
    candidate: &mut Candidate,
    glyphs: &[Glyph],
    divider: f32,
) -> Vec<(f32, f32)> {
    if candidate.ys.len() < 3 {
        return Vec::new();
    }
    let row_count = candidate.ys.len() - 1;
    let mut row_glyphs = vec![Vec::new(); row_count];
    for glyph in glyphs {
        if glyph.text.as_deref().map_or(true, |text| text.trim().is_empty()) {
            continue;
        }
        let Some((x, y)) = glyph_position(glyph) else {
            continue;
        };
        if x < candidate.xs[0] || x > *candidate.xs.last().unwrap() {
            continue;
        }
        if let Some(row) = coordinate_slot(&candidate.ys, y) {
            row_glyphs[row].push(CellGlyph::new(glyph));
        }
    }

    let mut boundaries = Vec::new();
    let mut heading_spans = Vec::new();
    for row in 1..row_count {
        let top = candidate.ys[row];
        let bottom = candidate.ys[row + 1];
        let mut starts: Vec<(f32, bool)> = Vec::new();
        let mut previous_baseline = None;
        let mut frequency_count = 0;
        let mut heading_count = 0;
        let mut preceding_line = false;
        for mut line in group_cell_lines(&row_glyphs[row]) {
            let Some(metrics) = line_metrics(&line) else {
                continue;
            };
            line.sort_by(|first, second| first.left.total_cmp(&second.left));
            let first = &line[0];
            let near_left = first.left <= candidate.xs[0] + (divider - candidate.xs[0]) * 0.35;
            let left_end = line.partition_point(|glyph| glyph.left < divider);
            let frequency = near_left
                && left_end > 0
                && frequency_label(&cell_text(&line[..left_end])).is_some();
            let heading = near_left
                && !frequency
                && matches!(first.glyph.font_weight, Some(700..=900));
            if frequency || heading {
                if starts.is_empty() && previous_baseline.is_some() {
                    preceding_line = true;
                }
                if !(heading && starts.last().is_some_and(|(_, was_heading)| *was_heading)) {
                    let start = previous_baseline
                        .map_or(top, |previous| (previous + metrics.baseline) * 0.5);
                    starts.push((start, heading));
                    frequency_count += usize::from(frequency);
                    heading_count += usize::from(heading);
                }
            }
            previous_baseline = Some(metrics.baseline);
        }
        if preceding_line || frequency_count < 2 || heading_count == 0 {
            continue;
        }
        for index in 0..starts.len() {
            let (start, heading) = starts[index];
            let end = starts.get(index + 1).map_or(bottom, |next| next.0);
            if index > 0 && start > top + COORD_TOLERANCE_PT && start < bottom - COORD_TOLERANCE_PT {
                boundaries.push(start);
            }
            if heading {
                heading_spans.push((start, end));
            }
        }
    }

    for &coordinate in &boundaries {
        candidate.horizontal.push(RuleLine {
            axis: Axis::Horizontal,
            coordinate,
            spans: vec![(candidate.xs[0], *candidate.xs.last().unwrap())],
        });
    }
    candidate.ys.extend(boundaries);
    candidate.ys.sort_by(f32::total_cmp);
    heading_spans
}

fn reconstruct_grid(
    candidate: &Candidate,
    glyphs: &[Glyph],
) -> Option<Vec<Vec<Option<TableCell>>>> {
    let row_count = candidate.ys.len().checked_sub(1)?;
    let column_count = candidate.xs.len().checked_sub(1)?;
    if row_count == 0 || column_count == 0 || row_count * column_count > 20_000 {
        return None;
    }

    let mut cells = DisjointSet::new(row_count * column_count);
    let mut row_edges = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let top = candidate.ys[row];
        let bottom = candidate.ys[row + 1];
        let mut edges = candidate.xs.clone();
        for column in 0..column_count.saturating_sub(1) {
            let divider = candidate.xs[column + 1];
            // A row may use shifted rules and merge some global columns.
            let tolerance = ((divider - candidate.xs[column])
                .min(candidate.xs[column + 2] - divider)
                * 0.45)
                .max(INTERSECTION_TOLERANCE_PT);
            let nearest = candidate
                .vertical
                .iter()
                .filter(|rule| {
                    rule.axis == Axis::Vertical
                        && (rule.coordinate - divider).abs() <= tolerance
                        && coverage(rule, top, bottom) >= 0.72
                })
                .min_by(|first, second| {
                    (first.coordinate - divider)
                        .abs()
                        .total_cmp(&(second.coordinate - divider).abs())
                });
            if let Some(rule) = nearest {
                edges[column + 1] = rule.coordinate;
            } else {
                cells.union(row * column_count + column, row * column_count + column + 1);
            }
        }
        row_edges.push(edges);
    }
    for row in 0..row_count.saturating_sub(1) {
        let divider = candidate.ys[row + 1];
        for column in 0..column_count {
            let left = candidate.xs[column];
            let right = candidate.xs[column + 1];
            if !has_rule_coverage(
                &candidate.horizontal,
                divider,
                left,
                right,
                Axis::Horizontal,
            ) {
                cells.union(row * column_count + column, (row + 1) * column_count + column);
            }
        }
    }

    let mut group_indices = BTreeMap::new();
    let mut groups = Vec::<CellGroup<'_>>::new();
    for row in 0..row_count {
        for column in 0..column_count {
            let root = cells.find(row * column_count + column);
            let group_index = *group_indices.entry(root).or_insert_with(|| {
                groups.push(CellGroup::new(row, column));
                groups.len() - 1
            });
            groups[group_index].include(row, column);
        }
    }

    for glyph in glyphs {
        let Some(text) = glyph.text.as_deref() else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let Some((x, y)) = glyph_position(glyph) else {
            continue;
        };
        let Some(row) = coordinate_slot(&candidate.ys, y) else {
            continue;
        };
        let Some(column) = coordinate_slot(&row_edges[row], x) else {
            continue;
        };
        let root = cells.find(row * column_count + column);
        if let Some(&group_index) = group_indices.get(&root) {
            groups[group_index].glyphs.push(CellGlyph::new(glyph));
        }
    }

    let mut output = vec![vec![None; column_count]; row_count];
    for group in groups {
        let text = cell_text(&group.glyphs);
        if text.is_empty() {
            continue;
        }
        let bbox_pt = [
            (group.top_row..=group.bottom_row)
                .map(|row| row_edges[row][group.left_column])
                .fold(f32::INFINITY, f32::min),
            candidate.ys[group.top_row],
            (group.top_row..=group.bottom_row)
                .map(|row| row_edges[row][group.right_column + 1])
                .fold(f32::NEG_INFINITY, f32::max),
            candidate.ys[group.bottom_row + 1],
        ];
        let bold = cell_bold(&group.glyphs);
        output[group.top_row][group.left_column] = Some(TableCell {
            text,
            bbox_pt,
            bold,
        });
    }

    Some(output)
}

struct CellGroup<'a> {
    top_row: usize,
    bottom_row: usize,
    left_column: usize,
    right_column: usize,
    glyphs: Vec<CellGlyph<'a>>,
}

impl<'a> CellGroup<'a> {
    fn new(row: usize, column: usize) -> Self {
        Self {
            top_row: row,
            bottom_row: row,
            left_column: column,
            right_column: column,
            glyphs: Vec::new(),
        }
    }

    fn include(&mut self, row: usize, column: usize) {
        self.top_row = self.top_row.min(row);
        self.bottom_row = self.bottom_row.max(row);
        self.left_column = self.left_column.min(column);
        self.right_column = self.right_column.max(column);
    }
}

impl<'a> CellGlyph<'a> {
    fn new(glyph: &'a Glyph) -> CellGlyph<'a> {
        let usable_bounds = glyph
            .bbox_pt
            .filter(|bounds| bounds[0] != bounds[2] || bounds[1] != bounds[3]);
        let baseline = glyph
            .origin_pt
            .map_or_else(|| glyph.bbox_pt.map_or(0.0, |bbox| bbox[3]), |origin| origin[1]);
        let (left, right, top, bottom) = usable_bounds.map_or_else(
            || {
                let x = glyph.origin_pt.map_or(0.0, |origin| origin[0]);
                (x, x, baseline, baseline)
            },
            |bbox| {
                (
                    bbox[0].min(bbox[2]),
                    bbox[0].max(bbox[2]),
                    bbox[1].min(bbox[3]),
                    bbox[1].max(bbox[3]),
                )
            },
        );
        Self {
            glyph,
            left,
            right,
            top,
            bottom,
            baseline,
            height: bottom - top,
            superscript: false,
        }
    }
}

fn glyph_position(glyph: &Glyph) -> Option<(f32, f32)> {
    if let Some(bbox) = glyph.bbox_pt {
        let left = bbox[0].min(bbox[2]);
        let right = bbox[0].max(bbox[2]);
        let top = bbox[1].min(bbox[3]);
        let bottom = bbox[1].max(bbox[3]);
        if right > left || bottom > top {
            return Some(((left + right) * 0.5, (top + bottom) * 0.5));
        }
    }
    glyph.origin_pt.map(|[x, y]| (x, y))
}

fn coordinate_slot(edges: &[f32], coordinate: f32) -> Option<usize> {
    if edges.len() < 2
        || coordinate < edges[0] - INTERSECTION_TOLERANCE_PT
        || coordinate > edges[edges.len() - 1] + INTERSECTION_TOLERANCE_PT
    {
        return None;
    }
    let coordinate = coordinate.clamp(edges[0], edges[edges.len() - 1]);
    let index = edges.partition_point(|edge| *edge <= coordinate).saturating_sub(1);
    Some(index.min(edges.len() - 2))
}

fn has_rule_coverage(
    rules: &[RuleLine],
    coordinate: f32,
    start: f32,
    end: f32,
    axis: Axis,
) -> bool {
    rules.iter().any(|line| {
        line.axis == axis
            && (line.coordinate - coordinate).abs() <= INTERSECTION_TOLERANCE_PT
            && coverage(line, start, end) >= 0.72
    })
}

fn coverage(line: &RuleLine, start: f32, end: f32) -> f32 {
    let length = end - start;
    if length <= 0.0 {
        return 0.0;
    }
    let mut covered = 0.0;
    for (span_start, span_end) in &line.spans {
        covered += (span_end.min(end) - span_start.max(start)).max(0.0);
    }
    (covered / length).clamp(0.0, 1.0)
}


fn find_header(
    rows: &[Vec<Option<TableCell>>],
    column_edges: &[f32],
) -> Result<Option<Header>, &'static str> {
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let labels: Vec<String> = row
        .iter()
        .map(|cell| cell.as_ref().map_or_else(String::new, |cell| clean_text(&cell.text)))
        .collect();
    if labels.is_empty() || labels.iter().all(String::is_empty) {
        return Ok(None);
    }
    if row.len() == 1
        && !is_inline_frequency_header(&labels[0])
        && row[0]
            .as_ref()
            .is_some_and(|cell| has_inline_frequency_entries(&cell.text))
    {
        return Ok(Some(Header {
            row: 0,
            kind: TableKind::Inline,
            frequency_labels: vec![None; labels.len()],
        }));
    }


    if frequency_label(&labels[0]).is_some() {
        let mut frequency_labels = vec![None; labels.len()];
        for cell in row.iter().flatten().filter(|cell| !cell.text.trim().is_empty()) {
            let label = frequency_label(&cell.text)
                .ok_or("grid frequency header contains an unrecognized column")?;
            assign_frequency_label(&mut frequency_labels, cell, column_edges, label)?;
        }
        if frequency_labels.iter().any(Option::is_none) {
            return Err("grid frequency header has a blank column not covered by a merged header");
        }
        return Ok(Some(Header {
            row: 0,
            kind: TableKind::Grid,
            frequency_labels,
        }));
    }

    if is_inline_frequency_header(&labels[0]) {
        return Ok(Some(Header {
            row: 0,
            kind: TableKind::Inline,
            frequency_labels: vec![None; labels.len()],
        }));
    }

    Ok(Some(Header {
        row: 0,
        kind: TableKind::Labelled,
        frequency_labels: vec![None; labels.len()],
    }))
}

fn is_inline_frequency_header(text: &str) -> bool {
    let normalized = clean_text(&normalize(text));
    let system_column = normalized.contains("system organ class")
        || normalized.contains("κατηγορια οργανικου συστηματος");
    let frequency_column =
        normalized.contains("frequency") || normalized.contains("συχνοτητα");
    let reaction_column = normalized.contains("preferred term")
        || normalized.contains("adverse reaction")
        || normalized.contains("adverse event")
        || (normalized.contains("προτιμωμεν") && normalized.contains("ορο"))
        || (normalized.contains("ανεπιθυμητ") && normalized.contains("ενεργει"));

    system_column && frequency_column && reaction_column
}

fn inline_frequency_entry(line: &str) -> Option<(String, &str)> {
    let (prefix, reaction) = line.split_once(':')?;
    Some((frequency_label(prefix)?, reaction.trim()))
}

fn has_inline_frequency_entries(text: &str) -> bool {
    text.lines().any(|line| inline_frequency_entry(line).is_some())
}

fn assign_frequency_label(
    labels: &mut [Option<String>],
    cell: &TableCell,
    column_edges: &[f32],
    label: String,
) -> Result<(), &'static str> {
    let columns = cell_columns(cell, column_edges);
    if columns.is_empty() {
        return Err("could not map a merged frequency header to table columns");
    }
    for column in columns {
        if let Some(previous) = &labels[column] {
            if previous != &label {
                return Err("overlapping frequency headers disagree about a merged column");
            }
        } else {
            labels[column] = Some(label.clone());
        }
    }
    Ok(())
}

fn cell_columns(cell: &TableCell, column_edges: &[f32]) -> Vec<usize> {
    if column_edges.len() < 2 {
        return Vec::new();
    }
    let left = cell.bbox_pt[0].min(cell.bbox_pt[2]);
    let right = cell.bbox_pt[0].max(cell.bbox_pt[2]);
    (0..column_edges.len() - 1)
        .filter(|&column| {
            let center = (column_edges[column] + column_edges[column + 1]) * 0.5;
            center >= left - COORD_TOLERANCE_PT && center <= right + COORD_TOLERANCE_PT
        })
        .collect()
}

fn frequency_label(text: &str) -> Option<String> {
    let normalized = clean_text(&normalize(text));
    frequency_label_matches(&normalized).then(|| clean_text(text))
}

fn frequency_label_matches(normalized: &str) -> bool {
    FREQUENCY_LABELS.iter().any(|label| {
        normalized
            .strip_prefix(label)
            .is_some_and(frequency_suffix_is_valid)
    })
}

fn contains_frequency_label(text: &str) -> bool {
    let normalized = clean_text(&normalize(text));
    let mut previous_is_word = false;
    for (index, character) in normalized.char_indices() {
        if !previous_is_word && frequency_label_matches(&normalized[index..]) {
            return true;
        }
        previous_is_word = character.is_alphanumeric();
    }
    false
}

fn is_unstructured_single_cell_box(grid: &[Vec<Option<TableCell>>]) -> bool {
    grid.len() == 1
        && grid[0].len() == 1
        && grid[0][0]
            .as_ref()
            .map_or(true, |cell| !contains_frequency_label(&cell.text))
}

fn frequency_suffix_is_valid(suffix: &str) -> bool {
    suffix.trim_start().chars().next().map_or(true, |character| {
        character.is_ascii_digit()
            || matches!(
                character,
                '(' | '≥' | '>' | '≤' | '<' | '=' | '-' | '–' | '—' | ':'
            )
    })
}



fn normalize(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .map(|character| match character {
            'ά' | 'ὰ' | 'ᾶ' => 'α',
            'έ' | 'ὲ' => 'ε',
            'ή' | 'ὴ' | 'ῆ' => 'η',
            'ί' | 'ὶ' | 'ῖ' | 'ϊ' | 'ΐ' | 'ῒ' | 'ῗ' => 'ι',
            'ό' | 'ὸ' => 'ο',
            'ύ' | 'ὺ' | 'ῦ' | 'ϋ' | 'ΰ' | 'ῢ' | 'ῧ' => 'υ',
            'ώ' | 'ὼ' | 'ῶ' => 'ω',
            other => other,
        })
        .filter(|character| !matches!(*character, '\u{0300}'..='\u{036f}'))
        .collect()
}

fn clean_text(text: &str) -> String {
    let mut cleaned = String::new();
    for part in text.split_whitespace() {
        if !cleaned.is_empty() {
            cleaned.push(' ');
        }
        cleaned.push_str(part);
    }
    cleaned
}

fn append_inline_text(target: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push(' ');
    }
    target.push_str(text);
}

fn flush_inline_reaction(
    output: &mut Vec<String>,
    reaction: &mut String,
    frequency: &Option<String>,
) {
    if !reaction.is_empty() {
        if let Some(label) = frequency {
            output.push(format!("{reaction} - {label}"));
        }
        reaction.clear();
    }
}

fn flatten_rows(
    rows: &[Vec<Option<TableCell>>],
    header: &Header,
    page_number: usize,
    candidate: &Candidate,
) -> Vec<String> {
    let mut output = Vec::new();
    match header.kind {
        TableKind::Grid => {
            for (row_index, row) in rows.iter().enumerate().skip(header.row + 1) {
                let present: Vec<(usize, &TableCell)> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(column, cell)| {
                        cell.as_ref()
                            .filter(|cell| !cell.text.trim().is_empty())
                            .map(|cell| (column, cell))
                    })
                    .collect();
                if present.is_empty() {
                    continue;
                }

                if present.len() == 1 && present[0].0 == 0 {
                    let cell = present[0].1;
                    if cell.bold == Some(false) {
                        if let Some(label) = header.frequency_labels.first().and_then(Option::as_ref) {
                            output.push(format!("{} - {}", clean_text(&cell.text), label));
                        }
                    } else {
                        output.push(clean_text(&cell.text));
                    }
                    continue;
                }

                for (column, cell) in present {
                    if let Some(label) = header.frequency_labels.get(column).and_then(Option::as_ref) {
                        output.push(format!("{} - {}", clean_text(&cell.text), label));
                    } else {
                        eprintln!(
                            "table extractor: skipped ambiguous frequency cell on page {page_number}, row {} at bbox_pt={:?}",
                            row_index + 1,
                            candidate.bbox
                        );
                    }
                }
            }
        }
        TableKind::Inline => {
            let mut current_frequency: Option<String> = None;
            let mut reaction = String::new();
            let mut heading = String::new();
            let first_row_is_data = rows[header.row]
                .first()
                .and_then(Option::as_ref)
                .is_some_and(|cell| {
                    !is_inline_frequency_header(&cell.text)
                        && has_inline_frequency_entries(&cell.text)
                });
            let start = header.row + usize::from(!first_row_is_data);
            for row in rows.iter().skip(start) {
                let Some(cell) = row.first().and_then(Option::as_ref) else {
                    continue;
                };
                let first = cell.text.lines().find(|line| !line.trim().is_empty());
                let starts_heading = cell.bold == Some(true)
                    || (has_inline_frequency_entries(&cell.text)
                        && first.is_some_and(|line| {
                            inline_frequency_entry(line).is_none()
                                && frequency_label(line).is_none()
                        }));
                if starts_heading {
                    flush_inline_reaction(&mut output, &mut reaction, &current_frequency);
                    current_frequency = None;
                }

                for raw_line in cell.text.lines() {
                    let line = clean_text(raw_line);
                    if line.is_empty() {
                        continue;
                    }
                    let (label, text) = match inline_frequency_entry(&line) {
                        Some((label, text)) => (Some(label), Some(text)),
                        None => (frequency_label(&line), None),
                    };
                    if let Some(label) = label {
                        if !heading.is_empty() {
                            output.push(std::mem::take(&mut heading));
                        }
                        flush_inline_reaction(&mut output, &mut reaction, &current_frequency);
                        current_frequency = Some(label);
                        if let Some(text) = text {
                            append_inline_text(&mut reaction, text);
                        }
                    } else if current_frequency.is_some() {
                        append_inline_text(&mut reaction, &line);
                    } else {
                        append_inline_text(&mut heading, &line);
                    }
                }
            }
            if !heading.is_empty() {
                output.push(heading);
            }
            flush_inline_reaction(&mut output, &mut reaction, &current_frequency);
        }
        TableKind::Labelled => {
            let header_row = rows.get(header.row);
            let mut current_reaction = String::new();
            let mut current_frequency: Option<String> = None;
            for (row_index, row) in rows.iter().enumerate().skip(header.row + 1) {
                let reaction = row
                    .first()
                    .and_then(Option::as_ref)
                    .map(|cell| clean_text(&cell.text))
                    .filter(|text| !text.is_empty());
                let row_frequency = reaction
                    .as_deref()
                    .filter(|text| !text.contains(':'))
                    .and_then(frequency_label);
                if row_frequency.is_none() {
                    if let Some(reaction) = &reaction {
                        current_reaction.clone_from(reaction);
                        if row
                            .first()
                            .and_then(Option::as_ref)
                            .is_some_and(|cell| cell.bold == Some(true))
                        {
                            current_frequency = None;
                        }
                    }
                }

                let values: Vec<(usize, &TableCell)> = row
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter_map(|(column, cell)| {
                        cell.as_ref()
                            .filter(|cell| !cell.text.trim().is_empty())
                            .map(|cell| (column, cell))
                    })
                    .collect();
                if let Some(label) = row_frequency.as_deref() {
                    current_frequency = Some(label.to_owned());
                    for (_, value) in &values {
                        output.push(format!("{} - {label}", clean_text(&value.text)));
                    }
                    continue;
                }
                if reaction.is_none() && !values.is_empty() {
                    if let Some(label) = &current_frequency {
                        for (_, value) in &values {
                            output.push(format!("{} - {label}", clean_text(&value.text)));
                        }
                        continue;
                    }
                }
                let reaction = match reaction {
                    Some(reaction) => reaction,
                    None if !current_reaction.is_empty() && !values.is_empty() => {
                        current_reaction.clone()
                    }
                    None if !values.is_empty() => {
                        eprintln!(
                            "table extractor: skipped labelled row {} without a first-column label on page {page_number} at bbox_pt={:?}",
                            row_index + 1,
                            candidate.bbox
                        );
                        continue;
                    }
                    None => continue,
                };

                if values.is_empty() {
                    output.push(reaction);
                } else if values.len() == 1 {
                    output.push(format!("{reaction} - {}", clean_text(&values[0].1.text)));
                } else {
                    let joined = values
                        .iter()
                        .map(|(column, value)| {
                            let label = header_row
                                .and_then(|row| row.get(*column))
                                .and_then(Option::as_ref)
                                .map_or_else(String::new, |cell| clean_text(&cell.text));
                            format!("{label}: {}", clean_text(&value.text))
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    output.push(format!("{reaction} - {joined}"));
                }
            }
        }
    }
    output
}

#[derive(Clone, Copy)]
struct LineMetrics {
    baseline: f32,
    top: f32,
    bottom: f32,
    font_height: f32,
}

fn line_metrics(line: &[CellGlyph<'_>]) -> Option<LineMetrics> {
    let mut baseline = 0.0;
    let mut count = 0usize;
    let mut top = f32::INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    let mut font_height = 0.0_f32;
    for glyph in line {
        if glyph
            .glyph
            .text
            .as_deref()
            .map_or(true, |text| text.trim().is_empty())
            || glyph.height <= 0.0
        {
            continue;
        }
        baseline += glyph.baseline;
        count += 1;
        top = top.min(glyph.top);
        bottom = bottom.max(glyph.bottom);
        font_height = font_height.max(glyph.height);
    }
    (count > 0).then_some(LineMetrics {
        baseline: baseline / count as f32,
        top,
        bottom,
        font_height,
    })
}

fn superscript_attachment_gap(
    marker: &[CellGlyph<'_>],
    line: &[CellGlyph<'_>],
    font_height: f32,
) -> Option<f32> {
    let nonempty = |glyph: &&CellGlyph<'_>| {
        glyph
            .glyph
            .text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
    };
    let marker_left = marker
        .iter()
        .filter(nonempty)
        .map(|glyph| glyph.left)
        .min_by(f32::total_cmp)?;
    let marker_right = marker
        .iter()
        .filter(nonempty)
        .map(|glyph| glyph.right)
        .max_by(f32::total_cmp)?;
    let line_left = line
        .iter()
        .filter(nonempty)
        .map(|glyph| glyph.left)
        .min_by(f32::total_cmp)?;
    let overlap_tolerance = (font_height * 0.35).max(1.0);
    let previous_right = line
        .iter()
        .filter(nonempty)
        .map(|glyph| glyph.right)
        .filter(|right| *right <= marker_left + overlap_tolerance)
        .max_by(f32::total_cmp);

    if let Some(previous_right) = previous_right {
        Some(marker_left - previous_right)
    } else {
        let gap = line_left - marker_right;
        (gap >= -overlap_tolerance).then_some(gap)
    }
}

// Only elevated small glyph lines can attach as superscripts.
fn attach_superscript_lines(lines: &mut Vec<Vec<CellGlyph<'_>>>) {
    let mut source_index = 0;
    while source_index < lines.len() {
        let Some(source_metrics) = line_metrics(&lines[source_index]) else {
            source_index += 1;
            continue;
        };
        let mut best_target: Option<(usize, f32)> = None;
        for (target_index, target_line) in lines.iter().enumerate() {
            if target_index == source_index {
                continue;
            }
            let Some(target_metrics) = line_metrics(target_line) else {
                continue;
            };
            if source_metrics.font_height >= target_metrics.font_height * 0.85 {
                continue;
            }

            let baseline_distance = target_metrics.baseline - source_metrics.baseline;
            if baseline_distance <= 0.0
                || baseline_distance > (target_metrics.font_height * 0.7).max(2.4)
            {
                continue;
            }
            let overlap = (source_metrics.bottom.min(target_metrics.bottom)
                - source_metrics.top.max(target_metrics.top))
            .max(0.0);
            let overlap_ratio =
                overlap / source_metrics.font_height.min(target_metrics.font_height);
            if overlap_ratio < 0.15 && baseline_distance > target_metrics.font_height * 0.4 {
                continue;
            }

            let Some(gap) = superscript_attachment_gap(
                &lines[source_index],
                target_line,
                target_metrics.font_height,
            ) else {
                continue;
            };
            if gap < -(target_metrics.font_height * 0.35).max(1.0)
                || gap > (target_metrics.font_height * 0.45).max(2.0)
            {
                continue;
            }

            let score = baseline_distance / target_metrics.font_height
                + gap.abs() / target_metrics.font_height
                + source_metrics.font_height / target_metrics.font_height
                - overlap_ratio.min(1.0) * 0.2;
            if best_target.map_or(true, |(_, best_score)| score < best_score) {
                best_target = Some((target_index, score));
            }
        }

        if let Some((target_index, _)) = best_target {
            let mut marker = lines.remove(source_index);
            for glyph in &mut marker {
                glyph.superscript = true;
            }
            let target_index = if target_index > source_index {
                target_index - 1
            } else {
                target_index
            };
            lines[target_index].append(&mut marker);
            if target_index == source_index {
                source_index += 1;
            }
        } else {
            source_index += 1;
        }
    }
}
fn cell_text(glyphs: &[CellGlyph<'_>]) -> String {
    if glyphs.is_empty() {
        return String::new();
    }

    let mut lines = group_cell_lines(glyphs);
    attach_superscript_lines(&mut lines);

    let mut text = String::new();
    for mut line in lines {
        if line.iter().all(|glyph| {
            glyph
                .glyph
                .text
                .as_deref()
                .map_or(true, |piece| piece.trim().is_empty())
        }) {
            continue;
        }
        while text
            .chars()
            .last()
            .is_some_and(char::is_whitespace)
        {
            text.pop();
        }
        if !text.is_empty() {
            text.push('\n');
        }

        line.sort_by(|first, second| {
            first
                .left
                .total_cmp(&second.left)
                .then_with(|| first.glyph.index.cmp(&second.glyph.index))
        });
        let font_height = line
            .iter()
            .map(|glyph| glyph.height)
            .filter(|height| *height > 0.0)
            .fold(0.0_f32, f32::max);
        let gap_threshold = (font_height * 0.16).max(1.0);
        let mut right_edge: Option<f32> = None;
        for glyph in line {
            let piece = glyph.glyph.text.as_deref().unwrap_or_default();
            if glyph.superscript {
                while text.chars().last().is_some_and(char::is_whitespace) {
                    text.pop();
                }
                if !piece.chars().all(char::is_whitespace) {
                    text.push_str(piece);
                    right_edge = Some(glyph.right);
                }
                continue;
            }
            if piece.chars().all(|character| character.is_whitespace()) {
                if !text.is_empty()
                    && !text.chars().last().is_some_and(char::is_whitespace)
                {
                    text.push(' ');
                }
                right_edge = Some(glyph.right);
                continue;
            }
            if let Some(previous_right) = right_edge {
                if glyph.left - previous_right > gap_threshold
                    && !text.is_empty()
                    && !text.chars().last().is_some_and(char::is_whitespace)
                {
                    text.push(' ');
                }
            }
            text.push_str(piece);
            right_edge = Some(glyph.right);
        }
    }
    text
}

fn group_cell_lines<'a>(glyphs: &[CellGlyph<'a>]) -> Vec<Vec<CellGlyph<'a>>> {
    let mut ordered = glyphs.to_vec();
    ordered.sort_by(|first, second| {
        first
            .baseline
            .total_cmp(&second.baseline)
            .then_with(|| first.left.total_cmp(&second.left))
            .then_with(|| first.glyph.index.cmp(&second.glyph.index))
    });

    let mut lines: Vec<Vec<CellGlyph<'a>>> = Vec::new();
    for glyph in ordered {
        if let Some(last) = lines.last_mut() {
            let reference_baseline = last.first().map_or(glyph.baseline, |first| first.baseline);
            if (glyph.baseline - reference_baseline).abs() <= 2.4 {
                last.push(glyph);
                continue;
            }
        }
        lines.push(vec![glyph]);
    }
    lines
}

fn cell_bold(glyphs: &[CellGlyph<'_>]) -> Option<bool> {
    let first = glyphs
        .iter()
        .filter(|glyph| {
            glyph
                .glyph
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
        })
        .min_by(|first, second| {
            first
                .left
                .total_cmp(&second.left)
                .then_with(|| first.baseline.total_cmp(&second.baseline))
                .then_with(|| first.glyph.index.cmp(&second.glyph.index))
        })?;

    match first.glyph.font_weight {
        Some(100..=699) => Some(false),
        Some(700..=900) => Some(true),
        _ => None,
    }
}

fn report_skip(page_number: usize, candidate: &Candidate, reason: &str) {
    eprintln!(
        "table extractor: skipped page {page_number} candidate bbox_pt={:?}: {reason}",
        candidate.bbox
    );
}
fn report_table_diagnostic(page_number: usize, candidate: &Candidate, reason: &str) {
    eprintln!(
        "table extractor: retained page {page_number} candidate bbox_pt={:?} without flattened lines: {reason}",
        candidate.bbox
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_grids_with_aligned_rules_remain_separate() {
        for transpose in [false, true] {
            let mut lines = Vec::new();
            for y in [0.0, 10.0, 20.0, 40.0, 50.0, 60.0] {
                lines.push(RuleLine {
                    axis: Axis::Horizontal,
                    coordinate: y,
                    spans: vec![(0.0, 20.0)],
                });
            }
            for x in [0.0, 20.0] {
                lines.push(RuleLine {
                    axis: Axis::Vertical,
                    coordinate: x,
                    spans: vec![(0.0, 20.0), (40.0, 60.0)],
                });
            }
            for (x, span) in [(10.0, (0.0, 20.0)), (15.0, (40.0, 60.0))] {
                lines.push(RuleLine {
                    axis: Axis::Vertical,
                    coordinate: x,
                    spans: vec![span],
                });
            }
            if transpose {
                for line in &mut lines {
                    line.axis = match line.axis {
                        Axis::Horizontal => Axis::Vertical,
                        Axis::Vertical => Axis::Horizontal,
                    };
                }
            }
            lines.sort_unstable_by_key(|line| axis_order(line.axis));
            let boxes: Vec<_> = find_candidates(lines)
                .into_iter()
                .map(|candidate| candidate.bbox)
                .collect();
            let second = if transpose {
                [40.0, 0.0, 60.0, 20.0]
            } else {
                [0.0, 40.0, 20.0, 60.0]
            };
            assert_eq!(boxes, vec![[0.0, 0.0, 20.0, 20.0], second]);
        }
    }

    #[test]
    fn single_row_table_is_not_dropped() {
        let mut lines = Vec::new();
        for y in [0.0, 20.0] {
            lines.push(RuleLine {
                axis: Axis::Horizontal,
                coordinate: y,
                spans: vec![(0.0, 80.0)],
            });
        }
        for x in [0.0, 20.0, 40.0, 60.0, 80.0] {
            lines.push(RuleLine {
                axis: Axis::Vertical,
                coordinate: x,
                spans: vec![(0.0, 20.0)],
            });
        }
        let boxes: Vec<_> = find_candidates(lines)
            .into_iter()
            .map(|candidate| candidate.bbox)
            .collect();
        assert_eq!(boxes, vec![[0.0, 0.0, 80.0, 20.0]]);
    }
    #[test]
    fn row_rule_gaps_do_not_split_a_grid() {
        let mut lines = Vec::new();
        for y in [0.0, 10.0, 20.0] {
            lines.push(RuleLine {
                axis: Axis::Horizontal,
                coordinate: y,
                spans: vec![(0.0, 20.0)],
            });
        }
        for x in [0.0, 10.0, 20.0] {
            lines.push(RuleLine {
                axis: Axis::Vertical,
                coordinate: x,
                spans: vec![(3.12, 9.76), (13.12, 19.76)],
            });
        }

        let candidates = find_candidates(lines);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].bbox, [0.0, 0.0, 20.0, 20.0]);
    }

    #[test]
    fn isolated_vertical_rule_does_not_create_a_column() {
        let mut lines = Vec::new();
        for y in (0..=100).step_by(10) {
            lines.push(RuleLine {
                axis: Axis::Horizontal,
                coordinate: y as f32,
                spans: vec![(0.0, 100.0)],
            });
        }
        for x in [0.0, 20.0, 40.0, 60.0, 80.0, 100.0] {
            lines.push(RuleLine {
                axis: Axis::Vertical,
                coordinate: x,
                spans: vec![(0.0, 100.0)],
            });
        }
        lines.push(RuleLine {
            axis: Axis::Vertical,
            coordinate: 30.0,
            spans: vec![(40.0, 50.0)],
        });

        let candidates = find_candidates(lines);
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].xs,
            vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        );
    }

    fn test_glyph(
        index: usize,
        text: &str,
        bbox: [f32; 4],
        baseline: f32,
    ) -> Glyph {
        Glyph {
            index,
            text: Some(text.to_owned()),
            unicode_value: text.chars().next().map_or(0, |character| character as u32),
            generated: Some(false),
            font_weight: Some(400),
            origin_pt: Some([bbox[0], baseline]),
            bbox_pt: Some(bbox),
        }
    }

    #[test]
    fn transformed_path_points_are_page_relative() {
        let matrix = PdfMatrix::identity()
            .translate(PdfPoints::new(210.48), PdfPoints::new(733.68))
            .unwrap();

        let first = path_point_in_page(
            &matrix,
            (PdfPoints::new(0.0), PdfPoints::new(0.0)),
            842.04,
        );
        let second = path_point_in_page(
            &matrix,
            (PdfPoints::new(0.0), PdfPoints::new(-12.6)),
            842.04,
        );

        assert!((first.0 - 210.48).abs() < 0.01);
        assert!((first.1 - 108.36).abs() < 0.01);
        assert!((second.0 - 210.48).abs() < 0.01);
        assert!((second.1 - 120.96).abs() < 0.01);
    }

    #[test]
    fn lower_footnote_line_stays_after_the_word() {
        let word = "sulfonylurea";
        let mut glyphs = word
            .chars()
            .enumerate()
            .map(|(index, character)| {
                let left = index as f32 * 5.0;
                test_glyph(
                    index,
                    &character.to_string(),
                    [left, 0.0, left + 4.29, 15.0],
                    15.0,
                )
            })
            .collect::<Vec<_>>();
        glyphs.push(test_glyph(
            glyphs.len(),
            "a",
            [0.0, 12.0, 3.0, 21.0],
            21.0,
        ));
        let cell_glyphs = glyphs.iter().map(CellGlyph::new).collect::<Vec<_>>();

        assert_eq!(cell_text(&cell_glyphs), "sulfonylurea\na");
    }

    #[test]
    fn raised_marker_remains_attached_to_the_word() {
        let word = "Reaction";
        let mut glyphs = word
            .chars()
            .enumerate()
            .map(|(index, character)| {
                let left = index as f32 * 5.0;
                test_glyph(
                    index,
                    &character.to_string(),
                    [left, 0.0, left + 4.29, 15.0],
                    15.0,
                )
            })
            .collect::<Vec<_>>();
        let marker_left = word.len() as f32 * 5.0 - 0.71;
        glyphs.push(test_glyph(
            glyphs.len(),
            "a",
            [marker_left, 3.0, marker_left + 3.0, 10.0],
            11.0,
        ));
        let cell_glyphs = glyphs.iter().map(CellGlyph::new).collect::<Vec<_>>();

        assert_eq!(cell_text(&cell_glyphs), "Reactiona");
    }

    #[test]
    fn separates_adjacent_words_at_pdf_word_gap() {
        let glyphs = vec![
            test_glyph(0, "emiaa", [0.0, 0.0, 24.57, 15.0], 15.0),
            test_glyph(1, "when", [27.36, 0.0, 50.0, 15.0], 15.0),
        ];
        let cell_glyphs = glyphs.iter().map(CellGlyph::new).collect::<Vec<_>>();

        assert_eq!(cell_text(&cell_glyphs), "emiaa when");
    }

    #[test]
    fn boxed_prose_is_not_a_single_cell_table() {
        let single_cell = |text: &str| {
            vec![vec![Some(TableCell {
                text: text.to_owned(),
                bbox_pt: [0.0, 0.0, 100.0, 20.0],
                bold: None,
            })]]
        };

        assert!(is_unstructured_single_cell_box(&single_cell(
            "In trials, the frequency of bronchitis was uncommon."
        )));
        assert!(!is_unstructured_single_cell_box(&single_cell(
            "Infections: Πολύ συχνές: viral infection. Συχνές: cough."
        )));

        let mut multi_cell = single_cell("Narrative header");
        multi_cell[0].push(None);
        assert!(!is_unstructured_single_cell_box(&multi_cell));
    }

    #[test]
    fn unruled_top_rows_inside_both_outer_rails_are_included() {
        let candidates = find_candidates(vec![
            RuleLine {
                axis: Axis::Horizontal,
                coordinate: 10.0,
                spans: vec![(0.0, 20.0)],
            },
            RuleLine {
                axis: Axis::Horizontal,
                coordinate: 20.0,
                spans: vec![(0.0, 20.0)],
            },
            RuleLine {
                axis: Axis::Vertical,
                coordinate: 0.0,
                spans: vec![(0.0, 20.0)],
            },
            RuleLine {
                axis: Axis::Vertical,
                coordinate: 10.0,
                spans: vec![(10.0, 20.0)],
            },
            RuleLine {
                axis: Axis::Vertical,
                coordinate: 20.0,
                spans: vec![(0.0, 20.0)],
            },
        ]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].ys, vec![0.0, 10.0, 20.0]);
        assert_eq!(candidates[0].bbox, [0.0, 0.0, 20.0, 20.0]);
    }

    #[test]
    fn unruled_frequency_columns_follow_repeated_glyph_alignment() {
        let candidate = || {
            find_candidates(vec![
                RuleLine {
                    axis: Axis::Horizontal,
                    coordinate: 0.0,
                    spans: vec![(0.0, 200.0)],
                },
                RuleLine {
                    axis: Axis::Horizontal,
                    coordinate: 40.0,
                    spans: vec![(0.0, 200.0)],
                },
                RuleLine {
                    axis: Axis::Vertical,
                    coordinate: 0.0,
                    spans: vec![(0.0, 40.0)],
                },
                RuleLine {
                    axis: Axis::Vertical,
                    coordinate: 200.0,
                    spans: vec![(0.0, 40.0)],
                },
            ])
            .into_iter()
            .next()
            .unwrap()
        };
        let right_label_glyphs = vec![
            test_glyph(0, "Skin and tissue", [10.0, 5.0, 80.0, 15.0], 13.0),
            test_glyph(1, "Common: pruritus", [130.0, 5.0, 195.0, 15.0], 13.0),
            test_glyph(2, "disorders", [10.0, 20.0, 55.0, 30.0], 28.0),
            test_glyph(3, "Uncommon: rash", [130.0, 20.0, 195.0, 30.0], 28.0),
        ];
        let mut table = candidate();

        infer_unruled_text_column(&mut table, &right_label_glyphs);
        assert_eq!(table.xs, vec![0.0, 128.75, 200.0]);
        let grid = reconstruct_grid(&table, &right_label_glyphs).unwrap();
        assert_eq!(
            grid[0][0].as_ref().unwrap().text,
            "Skin and tissue\ndisorders"
        );
        assert_eq!(
            grid[0][1].as_ref().unwrap().text,
            "Common: pruritus\nUncommon: rash"
        );

        let left_label_glyphs = vec![
            test_glyph(0, "Common:", [20.0, 5.0, 50.0, 15.0], 13.0),
            test_glyph(1, "pruritus", [130.0, 5.0, 180.0, 15.0], 13.0),
            test_glyph(2, "Uncommon:", [20.0, 20.0, 70.0, 30.0], 28.0),
            test_glyph(3, "rash", [130.0, 20.0, 155.0, 30.0], 28.0),
        ];
        assert_eq!(
            infer_unruled_frequency_column(&candidate(), &left_label_glyphs),
            Some(128.75)
        );
    }

    #[test]
    fn inline_frequency_rows_associate_wrapped_terms() {
        let cell = |text: &str, bold| {
            Some(TableCell {
                text: text.to_owned(),
                bbox_pt: [0.0, 0.0, 100.0, 10.0],
                bold,
            })
        };
        let rows = vec![
            vec![cell(
                "System organ class\nFrequency\nPreferred term",
                Some(true),
            )],
            vec![cell(
                "Infections and infestations\nVery common\nInfectionsb",
                Some(true),
            )],
            vec![cell(
                "Blood and lymphatic system disorders\nVery common\nNeutropeniac\nCommon\nFebrile neutropenia",
                Some(true),
            )],
        ];
        assert!(is_inline_frequency_header(
            "Κατηγορία οργανικού συστήματος\nΣυχνότητα\nΠροτιμώμενος όροςα (PT)"
        ));

        let header = find_header(&rows, &[0.0, 100.0]).unwrap().unwrap();
        assert!(matches!(header.kind, TableKind::Inline));
        let candidate = Candidate {
            bbox: [0.0, 0.0, 100.0, 30.0],
            xs: vec![0.0, 100.0],
            ys: vec![0.0, 10.0, 20.0, 30.0],
            horizontal: Vec::new(),
            vertical: Vec::new(),
        };

        assert_eq!(
            flatten_rows(&rows, &header, 10, &candidate),
            vec![
                "Infections and infestations",
                "Infectionsb - Very common",
                "Blood and lymphatic system disorders",
                "Neutropeniac - Very common",
                "Febrile neutropenia - Common",
            ]
        );
    }

    #[test]
    fn headerless_inline_rows_keep_complete_wrapped_reactions() {
        let cell = |text: &str| {
            Some(TableCell {
                text: text.to_owned(),
                bbox_pt: [70.0, 56.0, 524.0, 184.0],
                bold: Some(false),
            })
        };
        let rows = vec![
            vec![cell(
                "Νεοπλάσματα καλοήθη, κακοήθη\nκαι μη καθορισμένα\nΣπάνιες: Λέμφωμα, μη-Hodgkin λέμφωμα,\nλευχαιμία, μελάνωμα.\nΜη γνωστής συχνότητας: Ηπατοσπληνικό λέμφωμα από T-κύτταρα\n(πρωτίστως σε εφήβους και νεαρούς ενήλικες).",
            )],
            vec![cell(
                "Διαταραχές του αίματος\nΣυχνές: Ουδετεροπενία, λευκοπενία.",
            )],
        ];
        let header = find_header(&rows, &[70.0, 524.0]).unwrap().unwrap();
        assert!(matches!(header.kind, TableKind::Inline));
        let candidate = Candidate {
            bbox: [70.0, 56.0, 524.0, 250.0],
            xs: vec![70.0, 524.0],
            ys: vec![56.0, 184.0, 250.0],
            horizontal: Vec::new(),
            vertical: Vec::new(),
        };

        assert_eq!(
            flatten_rows(&rows, &header, 17, &candidate),
            vec![
                "Νεοπλάσματα καλοήθη, κακοήθη και μη καθορισμένα",
                "Λέμφωμα, μη-Hodgkin λέμφωμα, λευχαιμία, μελάνωμα. - Σπάνιες",
                "Ηπατοσπληνικό λέμφωμα από T-κύτταρα (πρωτίστως σε εφήβους και νεαρούς ενήλικες). - Μη γνωστής συχνότητας",
                "Διαταραχές του αίματος",
                "Ουδετεροπενία, λευκοπενία. - Συχνές",
            ]
        );
    }

    #[test]
    fn frequency_first_rows_associate_the_reaction_cell() {
        let cell = |text: &str, bold| {
            Some(TableCell {
                text: text.to_owned(),
                bbox_pt: [0.0, 0.0, 100.0, 10.0],
                bold,
            })
        };
        let rows = vec![
            vec![
                cell("Category / system organ class", Some(true)),
                cell("Adverse reactions", Some(true)),
            ],
            vec![cell("Blood disorders", Some(true)), None],
            vec![cell("Common", Some(false)), cell("Rhinitis", Some(false))],
            vec![
                cell("Uncommon", Some(false)),
                cell("Neutropenia", Some(false)),
            ],
        ];
        let header = find_header(&rows, &[0.0, 100.0, 200.0])
            .unwrap()
            .unwrap();
        let candidate = Candidate {
            bbox: [0.0, 0.0, 200.0, 40.0],
            xs: vec![0.0, 100.0, 200.0],
            ys: vec![0.0, 10.0, 20.0, 30.0, 40.0],
            horizontal: Vec::new(),
            vertical: Vec::new(),
        };

        assert_eq!(
            flatten_rows(&rows, &header, 11, &candidate),
            vec![
                "Blood disorders",
                "Rhinitis - Common",
                "Neutropenia - Uncommon",
            ]
        );
    }

    #[test]
    fn shifted_row_rules_preserve_their_merged_columns() {
        // The middle row uses x=158,344,437 instead of the surrounding
        // x=156,255,326,433 and has one genuinely merged pair of columns.
        let horizontal = [0.0, 25.0, 50.0, 75.0]
            .into_iter()
            .map(|coordinate| RuleLine {
                axis: Axis::Horizontal,
                coordinate,
                spans: vec![(65.5, 530.0)],
            })
            .collect();
        let mut vertical = vec![
            RuleLine {
                axis: Axis::Vertical,
                coordinate: 65.5,
                spans: vec![(0.0, 75.0)],
            },
            RuleLine {
                axis: Axis::Vertical,
                coordinate: 530.0,
                spans: vec![(0.0, 75.0)],
            },
        ];
        for coordinate in [156.0, 255.0, 326.0, 433.0] {
            vertical.push(RuleLine {
                axis: Axis::Vertical,
                coordinate,
                spans: vec![(0.0, 25.0), (50.0, 75.0)],
            });
        }
        for coordinate in [158.45, 344.23, 437.14] {
            vertical.push(RuleLine {
                axis: Axis::Vertical,
                coordinate,
                spans: vec![(25.0, 50.0)],
            });
        }
        let candidate = Candidate {
            bbox: [65.5, 0.0, 530.0, 75.0],
            xs: vec![65.5, 156.0, 255.0, 326.0, 433.0, 530.0],
            ys: vec![0.0, 25.0, 50.0, 75.0],
            horizontal,
            vertical,
        };
        let glyphs = vec![
            test_glyph(0, "Reaction", [71.0, 30.0, 110.0, 38.0], 38.0),
            test_glyph(1, "end", [156.8, 30.0, 157.8, 38.0], 38.0),
            test_glyph(2, "Unknown", [330.0, 30.0, 342.0, 38.0], 38.0),
            test_glyph(3, "0%", [350.0, 30.0, 360.0, 38.0], 38.0),
            test_glyph(4, "0%", [445.0, 30.0, 455.0, 38.0], 38.0),
        ];

        let grid = reconstruct_grid(&candidate, &glyphs).unwrap();
        assert_eq!(grid[1][0].as_ref().unwrap().text, "Reaction end");
        assert_eq!(grid[1][1].as_ref().unwrap().text, "Unknown");
        assert!(grid[1][2].is_none());
        assert_eq!(grid[1][1].as_ref().unwrap().bbox_pt[2], 344.23);
        assert_eq!(grid[1][3].as_ref().unwrap().text, "0%");
        assert_eq!(grid[1][4].as_ref().unwrap().text, "0%");
    }

    #[test]
    fn unruled_frequency_rows_preserve_full_width_bold_headings() {
        let mut candidate = Candidate {
            bbox: [92.9, 77.4, 519.3, 161.4],
            xs: vec![92.9, 519.3],
            ys: vec![77.4, 101.7, 161.4],
            horizontal: [77.4, 101.7, 161.4]
                .into_iter()
                .map(|coordinate| RuleLine {
                    axis: Axis::Horizontal,
                    coordinate,
                    spans: vec![(92.9, 519.3)],
                })
                .collect(),
            vertical: Vec::new(),
        };
        let mut glyphs = vec![
            test_glyph(0, "System Organ Class", [97.0, 80.0, 194.0, 89.0], 89.0),
            test_glyph(1, "Adverse drug reactions", [227.1, 80.0, 340.0, 89.0], 89.0),
            test_glyph(2, "Infections and infestations", [97.0, 104.5, 220.0, 111.7], 111.5),
            test_glyph(3, "Common", [97.1, 116.5, 128.3, 123.6], 123.5),
            test_glyph(4, "Nasopharyngitis", [227.1, 116.5, 299.0, 123.6], 123.5),
            test_glyph(5, "Blood and lymphatic", [97.2, 128.2, 223.0, 135.4], 135.3),
            test_glyph(6, "system disorders", [229.0, 128.2, 340.9, 135.4], 135.3),
            test_glyph(7, "Uncommon", [97.0, 140.1, 145.4, 147.4], 147.2),
            test_glyph(8, "Neutropenia", [227.1, 140.1, 292.8, 147.4], 147.2),
        ];
        for index in [0, 1, 2, 5, 6] {
            glyphs[index].font_weight = Some(700);
        }

        infer_unruled_text_column(&mut candidate, &glyphs);
        let grid = reconstruct_grid(&candidate, &glyphs).unwrap();
        assert_eq!(grid.len(), 5);
        assert_eq!(
            grid[3][0].as_ref().unwrap().text,
            "Blood and lymphatic system disorders"
        );
        assert!(grid[3][1].is_none());
        let header = find_header(&grid, &candidate.xs).unwrap().unwrap();
        assert_eq!(
            flatten_rows(&grid, &header, 11, &candidate),
            vec![
                "Infections and infestations",
                "Nasopharyngitis - Common",
                "Blood and lymphatic system disorders",
                "Neutropenia - Uncommon",
            ]
        );
    }

    #[test]
    fn right_aligned_inline_labels_are_not_cut_into_columns() {
        let candidate = Candidate {
            bbox: [70.0, 56.0, 524.0, 200.0],
            xs: vec![70.0, 524.0],
            ys: vec![56.0, 200.0],
            horizontal: Vec::new(),
            vertical: Vec::new(),
        };
        let glyphs = vec![
            test_glyph(0, "Σπάνιες", [193.0, 70.0, 228.0, 78.0], 78.0),
            test_glyph(1, ":", [229.0, 70.0, 231.0, 78.0], 78.0),
            test_glyph(2, "Λέμφωμα", [242.0, 70.0, 290.0, 78.0], 78.0),
            test_glyph(3, "Μη", [118.0, 90.0, 130.0, 98.0], 98.0),
            test_glyph(4, "γνωστής", [135.0, 90.0, 170.0, 98.0], 98.0),
            test_glyph(5, "συχνότητας", [174.0, 90.0, 227.0, 98.0], 98.0),
            test_glyph(6, ":", [229.0, 90.0, 231.0, 98.0], 98.0),
            test_glyph(7, "Ηπατοσπληνικό", [242.0, 90.0, 320.0, 98.0], 98.0),
            test_glyph(8, "Σπάνιες", [193.0, 110.0, 228.0, 118.0], 118.0),
            test_glyph(9, ":", [229.0, 110.0, 231.0, 118.0], 118.0),
            test_glyph(10, "Αντίδραση", [242.0, 110.0, 300.0, 118.0], 118.0),
        ];

        assert_eq!(infer_unruled_frequency_column(&candidate, &glyphs), None);
    }

    #[test]
    fn ruled_continuation_keeps_the_following_item_in_its_source_cell() {
        let xs = vec![70.9, 135.84, 203.28, 296.76, 363.0];
        let ys = vec![56.88, 108.0, 184.44, 248.16];
        let candidate = Candidate {
            bbox: [xs[0], ys[0], *xs.last().unwrap(), *ys.last().unwrap()],
            horizontal: ys
                .iter()
                .map(|&coordinate| RuleLine {
                    axis: Axis::Horizontal,
                    coordinate,
                    spans: vec![(xs[0], *xs.last().unwrap())],
                })
                .collect(),
            vertical: xs
                .iter()
                .map(|&coordinate| RuleLine {
                    axis: Axis::Vertical,
                    coordinate,
                    spans: vec![(ys[0], *ys.last().unwrap())],
                })
                .collect(),
            xs,
            ys,
        };
        let glyphs = vec![
            test_glyph(0, "προϊόντα", [208.6, 161.5, 248.0, 171.4], 169.1),
            test_glyph(1, "α", [249.2, 161.9, 252.4, 165.2], 165.1),
            test_glyph(2, "Μειωμένη όρεξη", [208.7, 174.4, 286.2, 181.9], 181.8),
            test_glyph(3, "Ζάλη", [208.7, 187.3, 234.8, 194.9], 194.9),
        ];

        let grid = reconstruct_grid(&candidate, &glyphs).unwrap();
        assert_eq!(
            grid[1][2].as_ref().unwrap().text,
            "προϊόνταα\nΜειωμένη όρεξη"
        );
        assert_eq!(grid[2][2].as_ref().unwrap().text, "Ζάλη");
    }

    #[test]
    fn raised_prefix_marker_stays_attached_to_the_word() {
        let glyphs = vec![
            test_glyph(0, "1", [203.88, 242.08, 207.36, 249.78], 249.78),
            test_glyph(
                1,
                "Tremor",
                [207.36, 242.4, 240.47, 254.62],
                254.62,
            ),
        ];
        let cell_glyphs = glyphs.iter().map(CellGlyph::new).collect::<Vec<_>>();

        assert_eq!(cell_text(&cell_glyphs), "1Tremor");
    }

    #[test]
    fn unruled_bottom_row_requires_matching_outer_rails() {
        for right_end in [10.0, 20.0] {
            let candidates = find_candidates(vec![
                RuleLine {
                    axis: Axis::Horizontal,
                    coordinate: 0.0,
                    spans: vec![(0.0, 20.0)],
                },
                RuleLine {
                    axis: Axis::Horizontal,
                    coordinate: 10.0,
                    spans: vec![(0.0, 20.0)],
                },
                RuleLine {
                    axis: Axis::Vertical,
                    coordinate: 0.0,
                    spans: vec![(0.0, 20.0)],
                },
                RuleLine {
                    axis: Axis::Vertical,
                    coordinate: 20.0,
                    spans: vec![(0.0, right_end)],
                },
            ]);
            if right_end == 20.0 {
                assert_eq!(candidates.len(), 1);
                assert_eq!(candidates[0].ys, vec![0.0, 10.0, 20.0]);
                assert_eq!(candidates[0].bbox, [0.0, 0.0, 20.0, 20.0]);
            } else {
                // A single protruding rule must not turn surrounding prose into a row.
                assert!(candidates.is_empty());
            }
        }
    }
}
