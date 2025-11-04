use dioxus::prelude::*;

#[component]
pub fn Search() -> Element {
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
            "."
        }
    }
}
