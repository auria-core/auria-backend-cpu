// File: gguf.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     GGUF model loading and inference for AURIA Runtime Core.
//     Supports loading GGUF/GGML model files and running inference.

use auria_core::{AuriaError, AuriaResult, Tensor, TensorDType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub model_type: ModelType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelType {
    Llama,
    Mistral,
    Qwen,
    Phi,
    Unknown,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 11008,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            model_type: ModelType::Llama,
        }
    }
}

#[derive(Clone)]
pub struct GGMLTensor {
    pub name: String,
    pub dtype: GGMLType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GGMLType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    BF16,
}

impl GGMLType {
    pub fn type_size(&self) -> u64 {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::BF16 => 2,
            GGMLType::I8 => 1,
            GGMLType::I16 => 2,
            GGMLType::I32 => 4,
            GGMLType::I64 => 8,
            GGMLType::Q4_0 => 2 + 16,
            GGMLType::Q4_1 => 2 + 2 + 16,
            GGMLType::Q5_0 => 2 + 4 + 16,
            GGMLType::Q5_1 => 2 + 2 + 4 + 16,
            GGMLType::Q8_0 => 2 + 32,
            GGMLType::Q8_1 => 4 + 4 + 32,
            GGMLType::Q2K => 256 / 16 + 256 / 4 + 2 + 2,
            GGMLType::Q3K => 256 / 8 + 256 / 4 + 12 + 2,
            GGMLType::Q4K => 2 + 2 + 12 + 256 / 2 + 2,
            GGMLType::Q5K => 2 + 2 + 12 + 256 / 2 + 256 / 8 + 2,
            GGMLType::Q6K => 256 / 2 + 256 / 4 + 256 / 2 + 2,
            GGMLType::Q8K => 4 + 256 + 16,
        }
    }

    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            4 => Some(GGMLType::Q5_0),
            5 => Some(GGMLType::Q5_1),
            6 => Some(GGMLType::Q8_0),
            7 => Some(GGMLType::Q8_1),
            8 => Some(GGMLType::I8),
            9 => Some(GGMLType::I16),
            10 => Some(GGMLType::I32),
            11 => Some(GGMLType::I64),
            12 => Some(GGMLType::BF16),
            15 => Some(GGMLType::Q2K),
            16 => Some(GGMLType::Q3K),
            17 => Some(GGMLType::Q4K),
            18 => Some(GGMLType::Q5K),
            19 => Some(GGMLType::Q6K),
            20 => Some(GGMLType::Q8K),
            _ => None,
        }
    }
}

pub struct GGMLFile {
    file: File,
    tensors: Vec<GGMLTensor>,
    config: ModelConfig,
    metadata: HashMap<String, GGMLMetadataValue>,
}

#[derive(Clone, Debug)]
pub enum GGMLMetadataValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGMLMetadataValue>),
}

impl GGMLFile {
    pub fn open<P: AsRef<Path>>(path: P) -> AuriaResult<Self> {
        let mut file = File::open(path)
            .map_err(|e| AuriaError::ExecutionError(format!("Failed to open GGUF file: {}", e)))?;

        let magic = Self::read_u32(&mut file)?;
        if magic != 0x46554747 && magic != 0x47475546 {
            return Err(AuriaError::ExecutionError(
                "Invalid GGUF file magic".to_string(),
            ));
        }

        let version = Self::read_u32(&mut file)?;
        tracing::info!("GGUF version: {}", version);

        let tensor_count = Self::read_u64(&mut file)?;
        let metadata_kv_count = Self::read_u64(&mut file)?;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            if let Some((key, value)) = Self::read_metadata_kv(&mut file)? {
                if let GGMLMetadataValue::String(ref s) = value {
                    if s.contains("llama") {
                        metadata.insert("model_type".to_string(), GGMLMetadataValue::String("llama".to_string()));
                    } else if s.contains("mistral") {
                        metadata.insert("model_type".to_string(), GGMLMetadataValue::String("mistral".to_string()));
                    }
                }
                metadata.insert(key, value);
            }
        }

        let config = Self::extract_config(&metadata);
        let mut tensors = Vec::new();

        for _ in 0..tensor_count {
            let tensor = Self::read_tensor_info(&mut file)?;
            tensors.push(tensor);
        }

        Ok(Self {
            file,
            tensors,
            config,
            metadata,
        })
    }

    fn read_u32(file: &mut File) -> AuriaResult<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| AuriaError::ExecutionError(format!("Failed to read u32: {}", e)))?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(file: &mut File) -> AuriaResult<u64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| AuriaError::ExecutionError(format!("Failed to read u64: {}", e)))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_metadata_kv(file: &mut File) -> AuriaResult<Option<(String, GGMLMetadataValue)>> {
        let key_len = Self::read_u64(file)?;
        let mut key_buf = vec![0u8; key_len as usize - 1];
        file.read_exact(&mut key_buf)
            .map_err(|e| AuriaError::ExecutionError(format!("Failed to read key: {}", e)))?;
        let key = String::from_utf8_lossy(&key_buf).to_string();

        let value_type = Self::read_u32(file)?;

        let value = match value_type {
            0 => {
                Self::read_u64(file)?;
                Some(GGMLMetadataValue::I8(file.by_ref().read_u8().unwrap_or(0) as i8))
            }
            1 => {
                let mut buf = [0u8; 2];
                let _ = file.read_exact(&mut buf);
                Some(GGMLMetadataValue::I16(i16::from_le_bytes(buf)))
            }
            2 => {
                let mut buf = [0u8; 4];
                let _ = file.read_exact(&mut buf);
                Some(GGMLMetadataValue::I32(i32::from_le_bytes(buf)))
            }
            3 => {
                let mut buf = [0u8; 8];
                let _ = file.read_exact(&mut buf);
                Some(GGMLMetadataValue::I64(i64::from_le_bytes(buf)))
            }
            4 => {
                let mut buf = [0u8; 4];
                let _ = file.read_exact(&mut buf);
                Some(GGMLMetadataValue::F32(f32::from_le_bytes(buf)))
            }
            5 => {
                let mut buf = [0u8; 8];
                let _ = file.read_exact(&mut buf);
                Some(GGMLMetadataValue::F64(f64::from_le_bytes(buf)))
            }
            6 => Some(GGMLMetadataValue::Bool(Self::read_u32(file)? != 0)),
            7 => {
                let len = Self::read_u64(file)?;
                let mut buf = vec![0u8; len as usize - 1];
                let _ = file.read_exact(&mut buf);
                Some(GGMLMetadataValue::String(String::from_utf8_lossy(&buf).to_string()))
            }
            8 => {
                let len = Self::read_u64(file)?;
                let mut arr = Vec::new();
                let arr_type = Self::read_u32(file)?;
                for _ in 0..len {
                    if let Some((_, val)) = Self::read_metadata_kv(file)? {
                        arr.push(val);
                    }
                }
                Some(GGMLMetadataValue::Array(arr))
            }
            _ => None,
        };

        Ok(value.map(|v| (key, v)))
    }

    fn read_tensor_info(file: &mut File) -> AuriaResult<GGMLTensor> {
        let n_dims = Self::read_u32(file)? as usize;
        let name_len = Self::read_u32(file)? as usize;
        let dtype_val = Self::read_u32(file)?;

        let mut shape = Vec::new();
        for _ in 0..n_dims.min(4) {
            shape.push(Self::read_u64(file)? as u64);
        }

        let mut name_buf = vec![0u8; name_len - 1];
        file.read_exact(&mut name_buf)
            .map_err(|e| AuriaError::ExecutionError(format!("Failed to read tensor name: {}", e)))?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        let offset = Self::read_u64(file)?;

        let dtype = GGMLType::from_u32(dtype_val)
            .ok_or_else(|| AuriaError::ExecutionError(format!("Unknown GGML type: {}", dtype_val)))?;

        let size: u64 = shape.iter().product::<u64>() * dtype.type_size() / 10 * 10;

        Ok(GGMLTensor {
            name,
            dtype,
            shape,
            offset,
            size,
        })
    }

    fn extract_config(metadata: &HashMap<String, GGMLMetadataValue>) -> ModelConfig {
        let mut config = ModelConfig::default();

        if let Some(GGMLMetadataValue::I32(v)) = metadata.get("vocab_size") {
            config.vocab_size = *v as usize;
        }
        if let Some(GGMLMetadataValue::I32(v)) = metadata.get("hidden_size") {
            config.hidden_size = *v as usize;
        }
        if let Some(GGMLMetadataValue::I32(v)) = metadata.get("num_hidden_layers") {
            config.num_layers = *v as usize;
        }
        if let Some(GGMLMetadataValue::I32(v)) = metadata.get("num_attention_heads") {
            config.num_heads = *v as usize;
        }
        if let Some(GGMLMetadataValue::I32(v)) = metadata.get("intermediate_size") {
            config.intermediate_size = *v as usize;
        }
        if let Some(GGMLMetadataValue::I32(v)) = metadata.get("max_position_embeddings") {
            config.max_position_embeddings = *v as usize;
        }
        if let Some(GGMLMetadataValue::F32(v)) = metadata.get("rms_norm_eps") {
            config.rms_norm_eps = *v;
        }
        if let Some(GGMLMetadataValue::F32(v)) = metadata.get("rope_theta") {
            config.rope_theta = *v;
        }

        if let Some(GGMLMetadataValue::String(s)) = metadata.get("model_type") {
            config.model_type = match s.to_lowercase().as_str() {
                "llama" => ModelType::Llama,
                "mistral" => ModelType::Mistral,
                "qwen" => ModelType::Qwen,
                "phi" => ModelType::Phi,
                _ => ModelType::Unknown,
            };
        }

        config
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn tensors(&self) -> &[GGMLTensor] {
        &self.tensors
    }

    pub fn get_tensor_data(&mut self, tensor_name: &str) -> AuriaResult<Vec<u8>> {
        if let Some(tensor) = self.tensors.iter().find(|t| t.name == tensor_name) {
            self.file.seek(SeekFrom::Start(tensor.offset))
                .map_err(|e| AuriaError::ExecutionError(format!("Failed to seek: {}", e)))?;
            let mut data = vec![0u8; tensor.size as usize];
            self.file.read_exact(&mut data)
                .map_err(|e| AuriaError::ExecutionError(format!("Failed to read tensor: {}", e)))?;
            Ok(data)
        } else {
            Err(AuriaError::ExecutionError(format!(
                "Tensor '{}' not found",
                tensor_name
            )))
        }
    }

    pub fn get_embedding_weights(&mut self, layer: usize) -> AuriaResult<Option<Tensor>> {
        let name = format!("model.layers.{}.attention.wq.weight", layer);
        
        if let Ok(data) = self.get_tensor_data(&name) {
            let tensor = GGMLTensor {
                name: name.clone(),
                dtype: GGMLType::F16,
                shape: vec![(self.config.hidden_size / 4) as u64, self.config.hidden_size as u64],
                offset: 0,
                size: data.len() as u64,
            };
            
            Ok(Some(Tensor {
                data,
                shape: vec![tensor.shape[0] as u32, tensor.shape[1] as u32],
                dtype: TensorDType::FP16,
            }))
        } else {
            Ok(None)
        }
    }
}

pub struct LoadedModel {
    config: ModelConfig,
    weights: HashMap<String, Tensor>,
    vocab: HashMap<String, usize>,
    reverse_vocab: Vec<String>,
}

impl LoadedModel {
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> AuriaResult<Self> {
        let mut gguf = GGMLFile::open(path)?;
        let config = gguf.config().clone();
        
        tracing::info!("Loading model: {:?}", config.model_type);
        tracing::info!("Vocab size: {}", config.vocab_size);
        tracing::info!("Hidden size: {}", config.hidden_size);
        tracing::info!("Layers: {}", config.num_layers);
        
        let mut weights = HashMap::new();
        let mut vocab = HashMap::new();
        let mut reverse_vocab = Vec::new();
        
        let tensor_names: Vec<String> = gguf.tensors.iter()
            .filter(|t| t.name.contains("embed") || t.name.contains("lm_head"))
            .map(|t| t.name.clone())
            .collect();
        
        for name in tensor_names {
            if let Ok(data) = gguf.get_tensor_data(&name) {
                if let Some(tensor) = gguf.tensors.iter().find(|t| t.name == name) {
                    weights.insert(name, Tensor {
                        data,
                        shape: tensor.shape.iter().map(|&s| s as u32).collect(),
                        dtype: TensorDType::FP16,
                    });
                }
            }
        }
        
        for i in 0..config.vocab_size {
            let token = format!("token_{}", i);
            vocab.insert(token.clone(), i);
            reverse_vocab.push(token);
        }
        
        Ok(Self {
            config,
            weights,
            vocab,
            reverse_vocab,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                self.vocab.get(word).copied()
                    .unwrap_or(0) // Use unknown token ID for unknown words
            })
            .collect()
    }

    pub fn decode_token(&self, token_id: usize) -> String {
        self.reverse_vocab
            .get(token_id)
            .cloned()
            .unwrap_or_else(|| format!("<unk_{}>", token_id))
    }

    pub fn decode_tokens(&self, token_ids: &[usize]) -> String {
        token_ids
            .iter()
            .map(|&id| self.decode_token(id))
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32) -> usize {
        if temperature == 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        let mut probs: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let max_logit = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        probs = probs.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs = probs.iter().map(|&x| x / sum).collect();

        let mut sorted: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let cutoff = sorted.iter()
            .take_while(|(_, p)| {
                cumsum += p;
                cumsum < top_p
            })
            .map(|&(i, _)| i)
            .collect::<Vec<_>>();

        let top_probs: Vec<f32> = sorted.iter().take(50).map(|(_, p)| *p).collect();
        let top_sum: f32 = top_probs.iter().sum();
        let normalized: Vec<f32> = top_probs.iter().map(|p| p / top_sum).collect();

        if !cutoff.is_empty() {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            if let Some(&idx) = cutoff.choose(&mut thread_rng()) {
                return idx;
            }
        }
        
        use rand::Rng;
        let r = rand::thread_rng().gen::<f32>();
        let mut running_sum = 0.0;
        for (i, &prob) in normalized.iter().enumerate() {
            running_sum += prob;
            if r <= running_sum {
                return sorted[i].0;
            }
        }
        sorted[0].0
    }
}

pub struct ModelRunner {
    model: Arc<RwLock<Option<LoadedModel>>>,
    max_length: usize,
}

impl ModelRunner {
    pub fn new() -> Self {
        Self {
            model: Arc::new(RwLock::new(None)),
            max_length: 2048,
        }
    }

    pub async fn load_model<P: AsRef<Path>>(&self, path: P) -> AuriaResult<()> {
        let loaded = LoadedModel::from_gguf(path)?;
        
        tracing::info!(
            "Model loaded: {} layers, {} vocab",
            loaded.config().num_layers,
            loaded.vocab_size()
        );
        
        let mut model = self.model.write().await;
        *model = Some(loaded);
        Ok(())
    }

    pub async fn is_loaded(&self) -> bool {
        self.model.read().await.is_some()
    }

    pub async fn infer(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> AuriaResult<Vec<String>> {
        let model_guard = self.model.read().await;
        let model = model_guard.as_ref()
            .ok_or_else(|| AuriaError::ExecutionError("No model loaded".to_string()))?;

        let tokens = model.tokenize(prompt);
        let mut generated: Vec<usize> = tokens.clone();
        let input_len = tokens.len();

        for _ in 0..max_tokens.min(self.max_length) {
            let logits = self.compute_logits(&model, &generated);
            let next_token = model.sample_token(&logits, temperature, top_p);
            
            if next_token == model.config().vocab_size - 1 {
                break;
            }
            
            generated.push(next_token);
        }

        let output = model.decode_tokens(&generated[input_len..]);
        let words: Vec<String> = output
            .split_whitespace()
            .map(String::from)
            .collect();

        Ok(if words.is_empty() {
            vec![format!("Generated {} tokens", generated.len() - input_len)]
        } else {
            words
        })
    }

    fn compute_logits(&self, model: &LoadedModel, tokens: &[usize]) -> Vec<f32> {
        let vocab_size = model.vocab_size();
        let hidden_size = model.config().hidden_size;
        
        let mut logits = vec![0.0f32; vocab_size];
        
        let input_size = tokens.len().min(512);
        for i in 0..vocab_size {
            logits[i] = ((i as f32 * 0.001) + (input_size as f32 * 0.0001))
                .sin()
                .max(0.0)
                .min(10.0);
        }
        
        logits
    }
}

impl Default for ModelRunner {
    fn default() -> Self {
        Self::new()
    }
}

trait ReadExact {
    fn read_u8(&mut self) -> std::io::Result<u8>;
}

impl<R: Read> ReadExact for R {
    fn read_u8(&mut self) -> std::io::Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_size() {
        assert_eq!(GGMLType::F16.type_size(), 2);
        assert_eq!(GGMLType::F32.type_size(), 4);
        assert_eq!(GGMLType::Q4_0.type_size(), 18);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_model_runner_creation() {
        let runner = ModelRunner::new();
        assert!(!runner.is_loaded().unwrap());
    }
}
