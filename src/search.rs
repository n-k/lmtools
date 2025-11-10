use dioxus::prelude::*;
use rusqlite::params;
use zerocopy::IntoBytes;

use crate::{
    lm::{
        get_cross_encoding_rank, get_embedding, get_embedding_model, get_llama_backend,
        get_reranking_model,
    },
    AppDb,
};

#[component]
pub fn Search() -> Element {
    let mut query = use_signal(|| "".to_string());
    let mut search_results: Signal<Vec<FTSResult>> = use_signal(|| vec![]);
    let mut status = use_resource(|| async move {
        let conn: crate::AppDb = consume_context();
        let scan_status = get_scan_status(conn).unwrap();

        scan_status
    });
    let search = move |q: String| async move {
        let conn: crate::AppDb = consume_context();
        let sr = match fts(conn, &q) {
            Err(e) => {
                eprintln!("{e:?}");
                vec![]
            }
            Ok(res) => res,
        };
        search_results.set(sr);
    };
    let scan_status = status.cloned().unwrap_or_default();
    let st = scan_status.to_percent();
    rsx! {
        div {
            style: "
            display: flex;
            flex-direction: column;
            class: main;
            margin: 1px;
            height: calc(100% - 4px);
            border: 1px solid red;
            ",
            div {
                style: "
                flex-grow: 0;
                display: flex;
                flex-direction: row;
                width: 100%;
                ",
                div {
                    style: "
                    flex-grow: 1;
                    display: flex;
                    flex-direction: row;
                    min-height: 1em;
                    width: 100%;
                    ",
                    div {style: "width: {st.error}%; background-color: red;", " "}
                    div {style: "width: {st.done}%; background-color: green;", " "}
                    div {style: "width: {st.scanning}%; background-color: blue;", " "}
                    div {style: "width: {st.pending}%; background-color: white;", " "}
                }
                button {
                    style: "flex-grow: 0;",
                    onclick: move |_| { status.restart(); },
                    "Refresh"
                }
            }
            div {
                style: "
                flex-grow: 0;
                display: flex;
                flex-direction: row;
                ",
                input {
                    style: "flex-grow: 1;",
                    value: query.cloned(),
                    oninput: move |e| { query.set(e.value()); },
                },
                button {
                    style: "flex-grow: 0;",
                    onclick: move |_| {
                        let q = query.cloned();
                        search_results.set(vec![]);
                        if q.is_empty() { return; }
                        spawn(async move {
                            search(q).await;
                        });
                    },
                    ">"
                }
            }
            div {
                style: "
                flex-grow: 1;
                overflow: auto;
                ",
                for r in search_results.cloned() {
                    div {
                        "{r.file_path} {r.chunk_index} {r.score}"
                        div {
                            style: "
                            font-size: 10px;
                            ",
                            "{r.chunk}"
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
struct FTSResult {
    file_path: String,
    chunk_index: usize,
    chunk: String,
    score: f32,
}

fn fts(conn: AppDb, query: &str) -> anyhow::Result<Vec<FTSResult>> {
    let mut results = vec![];
    let mut stmt = conn.prepare(
        r#"
        SELECT file_path, chunk_index, content, bm25(documents) AS score
        FROM documents
        WHERE documents MATCH ?1
        ORDER BY score
        LIMIT 10;
        "#,
    )?;
    let mut rows = stmt.query(params![query])?;
    while let Some(row) = rows.next()? {
        results.push(FTSResult {
            file_path: row.get(0)?,
            chunk_index: row.get(1)?,
            chunk: row.get(2)?,
            score: row.get(3)?,
        });
    }
    let backend = get_llama_backend();
    let embedding = {
        let embedding_model = get_embedding_model(backend)?;
        get_embedding(query, backend, &embedding_model)?
    };
    let mut stmt = conn.prepare(
        r#"
        SELECT file_path, chunk_index, content, distance
        FROM embeddings
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 10;
        "#,
    )?;
    let mut rows = stmt.query(params![embedding.as_bytes()])?;
    while let Some(row) = rows.next()? {
        results.push(FTSResult {
            file_path: row.get(0)?,
            chunk_index: row.get(1)?,
            chunk: row.get(2)?,
            score: row.get(3)?,
        });
    }
    let model = get_reranking_model(backend)?;
    for r in &mut results {
        let rank = get_cross_encoding_rank(query, &r.chunk, backend, &model)?;
        // println!("{:?}", rank);
        r.score = rank[0];
    }
    results.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or_else(|| std::cmp::Ordering::Equal)
    });
    results.reverse();

    Ok(results)
}

#[derive(Default, Clone)]
struct FilesScanStatus {
    pending: u64,
    scanning: u64,
    done: u64,
    error: u64,
}

impl FilesScanStatus {
    fn to_percent(&self) -> Self {
        let total = self.pending + self.scanning + self.done + self.error;
        let total = total as f64;
        Self {
            pending: (self.pending as f64 / total * 100.0) as u64,
            scanning: (self.scanning as f64 / total * 100.0) as u64,
            done: (self.done as f64 / total * 100.0) as u64,
            error: (self.error as f64 / total * 100.0) as u64,
        }
    }
}

fn get_scan_status(conn: AppDb) -> anyhow::Result<FilesScanStatus> {
    let mut stmt = conn.prepare(
        r#"
WITH all_statuses(status) AS (
  VALUES ('pending'), ('scanning'), ('done'), ('error')
)
SELECT
  a.status,
  COUNT(f.status) AS count
FROM
  all_statuses a
LEFT JOIN
  file_queue f ON f.status = a.status
GROUP BY
  a.status;
        "#,
    )?;
    let mut rows = stmt.query([])?;
    let mut scan_status = FilesScanStatus {
        pending: 0,
        scanning: 0,
        done: 0,
        error: 0,
    };
    while let Some(row) = rows.next()? {
        let status: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        let count: u64 = count as u64;

        match status.as_str() {
            "pending" => {
                scan_status.pending = count;
            }
            "scanning" => {
                scan_status.scanning = count;
            }
            "done" => {
                scan_status.done = count;
            }
            "error" => {
                scan_status.error = count;
            }
            _ => {}
        }
    }

    Ok(scan_status)
}
