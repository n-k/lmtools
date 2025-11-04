#![allow(non_snake_case)]
use std::rc::Rc;

use dioxus::prelude::*;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

use crate::workers::dir_scanner;

mod search;
mod workers;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");

pub type AppDb = Rc<PooledConnection<SqliteConnectionManager>>;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }

    let manager = SqliteConnectionManager::file("data.sqlite");
    let pool = Pool::builder()
        .max_size(10)
        .build(manager)?;

    {
        let conn = pool.get()?;
        conn.execute_batch(
        r#"
PRAGMA journal_mode=WAL;

-- Directories to scan
CREATE TABLE IF NOT EXISTS dir_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    status TEXT CHECK(status IN ('pending', 'scanning', 'done', 'error')) DEFAULT 'pending',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Files to index
CREATE TABLE IF NOT EXISTS file_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    dir_id INTEGER REFERENCES dir_queue(id) ON DELETE CASCADE,
    status TEXT CHECK(status IN ('pending', 'scanning', 'done', 'error')) DEFAULT 'pending',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    error TEXT
);

-- Actual full-text search table
CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
    file_path,
    chunk_index UNINDEXED,
    content
);

CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(embedding float[1024]);

INSERT OR IGNORE INTO dir_queue (path) VALUES ('/home/nk/stuff/code/nk/lmtools');
            "#
        )?;
    }

    {
        let conn = pool.get()?;
        // let pool = pool.clone();
        tokio::spawn(async move {
            if let Err(e) = dir_scanner(conn).await {
                eprint!("Error in dir scanner: {e:?}");
            }
        });
    }

    #[allow(deprecated)]
    LaunchBuilder::new()
        .with_context_provider(move || {
            Box::new(Rc::new(pool.get().unwrap()))
        })
        .launch(App);

    Ok(())
}

#[component]
fn App() -> Element {
    let _db: AppDb = consume_context();
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        Router::<Route> {}
    }
}

#[component]
fn Home() -> Element {
    search::Search()
}

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[layout(Navbar)]
    #[route("/")]
    Home {},
    #[route("/:..segments")]
    PageNotFound { segments: Vec<String> },
}

/// Shared navbar component.
#[component]
fn Navbar() -> Element {
    rsx! {
        div {
            id: "navbar",
            Link {
                to: Route::Home {},
                "Home"
            }
        }
        div {
            class: "main",
            Outlet::<Route> {}
        }
    }
}

/// 404 page component shown when a user navigates to an invalid route.
///
/// Displays an error message and provides a link back to the home page.
#[component]
fn PageNotFound(segments: Vec<String>) -> Element {
    rsx! {
        "Could not find the page you are looking for."
        Link { to: Route::Home {}, "Go To Home" }
    }
}
