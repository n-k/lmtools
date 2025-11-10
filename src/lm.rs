use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::{bail, Context};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel},
};

static LLAMA_CPP_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn get_llama_backend() -> &'static LlamaBackend {
    LLAMA_CPP_BACKEND.get_or_init(|| LlamaBackend::init().unwrap())
}

pub fn get_embedding_model(backend: &LlamaBackend) -> anyhow::Result<LlamaModel> {
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(
        &backend,
        "./models/all-minilm-l6-v2-q4_k_m.gguf",
        &model_params,
    )
    .with_context(|| "unable to load model")?;
    Ok(model)
}

pub fn get_reranking_model(backend: &LlamaBackend) -> anyhow::Result<LlamaModel> {
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(
        &backend,
        "./models/jina-reranker-v1-tiny-en.Q8_0.gguf",
        &model_params,
    )
    .with_context(|| "unable to load model")?;
    Ok(model)
}

pub fn tokenize_document_chunks(
    text: &str,
    backend: &LlamaBackend,
    model: &LlamaModel,
) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
    let stopwords: HashSet<&'static str> = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "is", "it", "that",
        "this", "with", "as", "at", "by", "from",
    ]
    .into_iter()
    .collect();

    // Split, normalize, filter out stopwords
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| !stopwords.contains(w.as_str()))
        .collect();
    let text = words.join(" ");
    let tokens = model.str_to_token(&text, llama_cpp_2::model::AddBos::Never)?;
    let chunks: Vec<_> = tokens.chunks(model.n_ctx_train() as usize).collect();

    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;
    let mut batch = LlamaBatch::new(model.n_ctx_train() as usize, 1);

    let mut results = vec![];
    for chunk in chunks {
        let s = model.tokens_to_str(chunk, llama_cpp_2::model::Special::Tokenize)?;

        batch.add_sequence(chunk, 0, false)?;
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;
        let embedding = ctx
            .embeddings_seq_ith(0)
            .with_context(|| "Failed to get embeddings")?;
        batch.clear();
        let embedding = normalize(embedding);

        results.push((s, embedding));
    }
    Ok(results)
}

pub fn get_cross_encoding_rank(
    query: &str,
    s: &str,
    backend: &LlamaBackend,
    model: &LlamaModel,
) -> anyhow::Result<Vec<f32>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true)
        .with_pooling_type(llama_cpp_2::context::params::LlamaPoolingType::Rank)
        .with_n_ubatch(model.n_ctx_train() / 2)
        .with_n_batch(model.n_ctx_train());
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let text = format!("{query}</s><s>{s}");
    let tokens = model.str_to_token(&text, llama_cpp_2::model::AddBos::Always)?;

    if tokens.len() > model.n_ctx_train() as usize {
        bail!("input longer than context length");
    }

    let mut batch = LlamaBatch::new(model.n_ctx_train() as usize, 1);
    batch.add_sequence(&tokens, 0, false)?;

    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;
    let embedding = ctx
        .embeddings_seq_ith(0)
        .with_context(|| "Failed to get embeddings")?;
    batch.clear();
    // let embedding = normalize(embedding);

    Ok(embedding.into())
}

pub fn get_embedding(
    s: &str,
    backend: &LlamaBackend,
    model: &LlamaModel,
) -> anyhow::Result<Vec<f32>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let tokens = model.str_to_token(s, llama_cpp_2::model::AddBos::Never)?;

    if tokens.len() > model.n_ctx_train() as usize {
        bail!("input longer than context length");
    }

    let mut batch = LlamaBatch::new(model.n_ctx_train() as usize, 1);
    batch.add_sequence(&tokens, 0, false)?;

    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;
    let embedding = ctx
        .embeddings_seq_ith(0)
        .with_context(|| "Failed to get embeddings")?;
    batch.clear();
    let embedding = normalize(embedding);

    Ok(embedding)
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}
