use dioxus::prelude::*;

use crate::AppDb;

#[component]
pub fn Search() -> Element {
    let mut query = use_signal(|| "".to_string());
    let mut status = use_resource(|| async move {
        let conn: crate::AppDb = consume_context();
        let scan_status = get_scan_status(conn).unwrap();

        scan_status
    });
    let mut search = use_resource(move || async move {
        let q = query.cloned();
        if q.is_empty() {
            return ();
        }
        query.set("".to_string());

        ()
    });
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
                display: flex;
                flex-direction: row;
                height: 1em;
                width: 100%;
                border: 1px solid green;
                ",
                div {style: "width: {st.error}%; background-color: red;", ""}
                div {style: "width: {st.done}%; background-color: green;", ""}
                div {style: "width: {st.scanning}%; background-color: blue;", ""}
                div {style: "width: {st.pending}%; background-color: white;", ""}
            }
            button {
                onclick: move |_| { status.restart(); },
                "Refresh Status"
            }
            input {
                value: query.cloned(),
                oninput: move |e| { query.set(e.value()); },
                onkeypress: move |e| {
                    if Key::Enter == e.data.key() {
                        search.restart();
                    }
                },
            }
        }
    }
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
