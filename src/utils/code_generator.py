import json
import os
import re

# Definições dos tokens especiais
PAD_TOKEN_STR = "<pad>"
SOS_TOKEN_STR = "<sos>"
EOS_TOKEN_STR = "<eos>"
UNK_TOKEN_STR = "<unk>"
SEP_TOKEN_STR = "<sep>"

class ArduinoCodeGenerator:
    """
    Generates Arduino-compatible C++ code (for both AVR and ARM architectures)
    from a Q&A model's JSON parameter files.
    """

    def __init__(self, output_dir="output_arduino_qna_code"):
        self.output_dir = output_dir
        self.params_data = {}
        self.vocab_list = []
        self.max_token_string_length = 1
        self.model_config = {}
        self.special_token_ids_map = {}

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    class _CppFormatter:
        """
        Helper functions to convert Python arrays into properly formatted C++ code strings.
        """

        @staticmethod
        def format_array(arr):
            """
            Formats a 1D array into a C++ float array string.
            """
            if not isinstance(arr, (list, tuple)):
                return "{}"
            return (
                '{' + ', '.join(f"{float(x):.8f}f" for x in arr if isinstance(x, (int, float))) + '}'
                if arr else '{}'
            )

        @staticmethod
        def format_2d_array(arr):
            """
            Formats a 2D array into a nested C++ float array string.
            Handles flat arrays as well (treated as single row).
            """
            if not isinstance(arr, (list, tuple)) or not arr:
                return '{}'

            if isinstance(arr[0], (list, tuple)):
                return '{' + ', '.join(
                    '{' + ', '.join(f"{float(x):.8f}f" for x in row) + '}' for row in arr
                ) + '}'
            elif all(isinstance(x, (int, float)) for x in arr):
                return '{' + ', '.join(f"{float(x):.8f}f" for x in arr) + '}'
            
            return '{}'

        @staticmethod
        def format_2d_array_bias_safe(arr_of_layers):
            """
            Formats a list of 1D arrays (or None) into a 2D C++ array.
            Used for optional bias parameters.
            """
            if not isinstance(arr_of_layers, (list, tuple)) or not arr_of_layers:
                return '{}'
            
            return '{' + ', '.join(
                '{' + ', '.join(f"{float(x):.8f}f" for x in layer) + '}' if layer is not None else '{}'
                for layer in arr_of_layers
            ) + '}'

        @staticmethod
        def format_3d_array(arr_of_layers):
            """
            Formats a 3D array into a nested C++ array string.
            """
            if not isinstance(arr_of_layers, (list, tuple)) or not arr_of_layers:
                return '{}'
            
            return '{' + ', '.join(
                ArduinoCodeGenerator._CppFormatter.format_2d_array(layer_data)
                for layer_data in arr_of_layers
            ) + '}'


    def _load_json_files(self, json_files):
        """
        Loads the specified JSON files into the 'params_data' dictionary.

        Args:
            json_files (list): List of file paths to JSON files.

        Returns:
            bool: True if all files were loaded successfully, False otherwise.
        """
        for file_name in json_files:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    base_name = os.path.basename(file_name)
                    self.params_data[base_name] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to decode JSON in file '{file_name}': {e}")
                return False
            except Exception as e:
                print(f"[ERROR] Failed to load file '{file_name}': {e}")
                return False

        return True


    def _validate_required_files(self):
        """
        Validates whether all required JSON files have been successfully loaded.

        Returns:
            bool: True if all required files are present, False otherwise.
        """
        required_files = [
            'final_norm.json',
            'vocab_list.json',
            'transformer_blocks.json',
            'embedding.json',
            'out_head.json',
            'config.json'
        ]

        for req_file in required_files:
            if req_file not in self.params_data:
                print(f"[ERROR] Required JSON file '{req_file}' not found.")
                print(f"[INFO] Available files: {list(self.params_data.keys())}")
                return False

        return True


    def _process_vocabulary(self):
        """
        Processes the vocabulary data from 'vocab_list.json' and populates `self.vocab_list`.
        Also computes the maximum token string length.

        Returns:
            bool: True if the vocabulary was processed successfully, False otherwise.
        """
        vocab_file_key = 'vocab_list.json'
        vocab_data_loaded = self.params_data.get(vocab_file_key)

        if isinstance(vocab_data_loaded, list):
            # Vocabulary is a list of tokens indexed by position
            self.vocab_list = vocab_data_loaded

        elif isinstance(vocab_data_loaded, dict):
            # Vocabulary is a dictionary mapping token string to ID
            max_id = max(vocab_data_loaded.values(), default=-1)
            self.vocab_list = [""] * (max_id + 1)

            for token, token_id in vocab_data_loaded.items():
                if 0 <= token_id < len(self.vocab_list):
                    self.vocab_list[token_id] = token
                else:
                    print(f"[WARNING] Token ID {token_id} is out of bounds for '{token}'. Skipping.")

            # Fill empty slots with unknown token
            for i in range(len(self.vocab_list)):
                if not self.vocab_list[i]:
                    self.vocab_list[i] = UNK_TOKEN_STR

        else:
            print(f"[ERROR] Unsupported format for '{vocab_file_key}'. Expected a list or dictionary.")
            return False

        # Compute the maximum token string length
        self.max_token_string_length = max(
            (len(str(token)) for token in self.vocab_list),
            default=1
        )

        return True


    def _extract_config_parameters(self):
        """
        Extracts and converts model configuration parameters from 'config.json' in params_data.

        Returns:
            bool: True if parameters were successfully extracted and converted, False otherwise.
        """
        config_data = self.params_data.get('config.json', {})

        try:
            self.model_config = {
                'n_heads': int(config_data.get('n_heads', 0)),
                'n_layers': int(config_data.get('n_layers', 0)),
                'emb_dim': int(config_data.get('emb_dim', 0)),
                'context_length': int(config_data.get('context_length', 0)),
                'vocab_size': int(config_data.get('vocab_size', 0)),
            }
        except (ValueError, TypeError) as e:
            print(f"[ERROR] Failed to convert configuration parameter to int: {e}")
            return False

        return True


    def _map_special_tokens(self):
        """
        Maps special token strings to their corresponding IDs based on `self.vocab_list`.
        Falls back to default IDs if tokens are not found.

        Returns:
            bool: Always returns True.
        """
        default_special_ids = {
            "SOS_TOKEN_ID": 0,
            "EOS_TOKEN_ID": 1,
            "PAD_TOKEN_ID": 2,
            "UNK_TOKEN_ID": 3,
            "SEP_TOKEN_ID": 4,
        }

        special_tokens = [
            ("SOS_TOKEN", SOS_TOKEN_STR),
            ("EOS_TOKEN", EOS_TOKEN_STR),
            ("PAD_TOKEN", PAD_TOKEN_STR),
            ("UNK_TOKEN", UNK_TOKEN_STR),
            ("SEP_TOKEN", SEP_TOKEN_STR),
        ]

        for token_name_enum, token_str_val in special_tokens:
            key = f"{token_name_enum}_ID"
            try:
                token_id = self.vocab_list.index(token_str_val)
                self.special_token_ids_map[key] = token_id
            except ValueError:
                default_id = default_special_ids.get(key, -1)
                self.special_token_ids_map[key] = default_id
                print(f"[WARNING] Special token '{token_str_val}' not found in vocabulary. Using default ID: {default_id}.")

        return True


    def _prepare_transformer_parameters(self):
        """
        Prepares and formats transformer block parameters for C++ code generation.

        Returns:
            dict: A mapping of C++ variable names to their formatted parameter strings.
        """
        n_layers = self.model_config.get('n_layers', 0)
        trf_params_str_map = {
            "trf_att_W_query_weight": "{}",
            "trf_att_W_key_weight": "{}",
            "trf_att_W_value_weight": "{}",
            "trf_att_out_proj_weight": "{}",
            "trf_att_out_proj_bias": "{}",
            "trf_ff_lin1_weight": "{}",
            "trf_ff_lin1_bias": "{}",
            "trf_ff_lin2_weight": "{}",
            "trf_ff_lin2_bias": "{}",
            "trf_norm1_scale": "{}",
            "trf_norm1_shift": "{}",
            "trf_norm2_scale": "{}",
            "trf_norm2_shift": "{}"
        }

        if n_layers > 0 and 'transformer_blocks.json' in self.params_data:
            transformer_blocks_data = self.params_data['transformer_blocks.json']
            is_dict = isinstance(transformer_blocks_data, dict)

            param_details = [
                ('att.W_query.weight', self._CppFormatter.format_3d_array, "trf_att_W_query_weight"),
                ('att.out_proj.bias', self._CppFormatter.format_2d_array_bias_safe, "trf_att_out_proj_bias"),
                ('att.W_key.weight', self._CppFormatter.format_3d_array, "trf_att_W_key_weight"),
                ('att.W_value.weight', self._CppFormatter.format_3d_array, "trf_att_W_value_weight"),
                ('att.out_proj.weight', self._CppFormatter.format_3d_array, "trf_att_out_proj_weight"),
                ('ff.lin1.weight', self._CppFormatter.format_3d_array, "trf_ff_lin1_weight"),
                ('ff.lin1.bias', self._CppFormatter.format_2d_array_bias_safe, "trf_ff_lin1_bias"),
                ('ff.lin2.weight', self._CppFormatter.format_3d_array, "trf_ff_lin2_weight"),
                ('ff.lin2.bias', self._CppFormatter.format_2d_array_bias_safe, "trf_ff_lin2_bias"),
                ('norm1.scale', self._CppFormatter.format_2d_array, "trf_norm1_scale"),
                ('norm1.shift', self._CppFormatter.format_2d_array, "trf_norm1_shift"),
                ('norm2.scale', self._CppFormatter.format_2d_array, "trf_norm2_scale"),
                ('norm2.shift', self._CppFormatter.format_2d_array, "trf_norm2_shift"),
            ]

            for param_json_name, formatter_func, cpp_var_name in param_details:
                data_list = []
                all_layers_valid = True

                for layer_idx in range(n_layers):
                    block_key = f'block_{layer_idx}' if is_dict else layer_idx
                    try:
                        block_data = transformer_blocks_data[block_key]
                        if not isinstance(block_data, dict):
                            all_layers_valid = False
                            break

                        # For bias parameters, allow missing keys and append None
                        if ".bias" in param_json_name and param_json_name not in block_data:
                            data_list.append(None)
                        else:
                            data_list.append(block_data[param_json_name])
                    except (KeyError, IndexError, TypeError) as e:
                        if ".bias" in param_json_name:
                            data_list.append(None)
                        else:
                            print(f"[WARNING] Issue accessing '{param_json_name}' in block '{block_key}': {e}. Using empty array for this parameter.")
                            all_layers_valid = False
                            break

                if all_layers_valid:
                    trf_params_str_map[cpp_var_name] = formatter_func(data_list)
                else:
                    trf_params_str_map[cpp_var_name] = "{}"

        return trf_params_str_map


    def _generate_header_defines(self):
        """
        Generates C++ #define directives for the Arduino header file,
        based on the loaded model configuration and special token mappings.

        Returns:
            str: Multiline string containing all #define statements.
        """
        vocab_size = max(1, self.model_config.get('vocab_size', 0))
        emb_dim = self.model_config.get('emb_dim', 0)
        context_length = max(1, self.model_config.get('context_length', 0))
        n_layers = self.model_config.get('n_layers', 0)
        n_heads = self.model_config.get('n_heads', 0)

        max_prompt_tokens_buffer_size = context_length * 2 if context_length > 0 else 10
        unk_token_id_val = self.special_token_ids_map.get("UNK_TOKEN_ID", -1)

        return f"""// Model Settings
    #define VOCAB_SIZE {vocab_size}
    #define EMB_DIM {emb_dim}
    #define CONTEXT_LENGTH {context_length}
    #define N_LAYERS {n_layers}
    #define N_HEADS {n_heads}
    #define HEAD_DIM (EMB_DIM / N_HEADS)

    // Special Tokens
    #define SOS_TOKEN_ID {self.special_token_ids_map.get('SOS_TOKEN_ID', -1)}
    #define EOS_TOKEN_ID {self.special_token_ids_map.get('EOS_TOKEN_ID', -1)}
    #define PAD_TOKEN_ID {self.special_token_ids_map.get('PAD_TOKEN_ID', -1)}
    #define UNK_TOKEN_ID {unk_token_id_val}
    #define SEP_TOKEN_ID {self.special_token_ids_map.get('SEP_TOKEN_ID', -1)}

    // Constants for buffers
    #define MAX_TOKEN_STRING_LENGTH {max(1, self.max_token_string_length)}
    #define MAX_PROMPT_TOKENS_BUFFER_SIZE {max_prompt_tokens_buffer_size}
    #define MAX_GENERATED_TEXT_LENGTH (CONTEXT_LENGTH * 5 + 100)
    """


    def _generate_header_weight_declarations(self):
        """Generates the weights declarations for the header file."""
        n_layers = self.model_config['n_layers']
        
        declarations = f"""
extern const float MODEL_DATA_STORAGE tok_emb_weight[VOCAB_SIZE][EMB_DIM];
extern const float MODEL_DATA_STORAGE pos_emb_weight[CONTEXT_LENGTH][EMB_DIM];
"""
        if n_layers > 0:
            declarations += f"""#if N_LAYERS > 0
extern const float MODEL_DATA_STORAGE trf_att_W_query_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_W_key_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_W_value_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_out_proj_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_out_proj_bias[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin1_weight[N_LAYERS][4*EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin1_bias[N_LAYERS][4*EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin2_weight[N_LAYERS][EMB_DIM][4*EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin2_bias[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm1_scale[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm1_shift[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm2_scale[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm2_shift[N_LAYERS][EMB_DIM];
#endif
"""
        declarations += f"""extern const float MODEL_DATA_STORAGE final_norm_scale[EMB_DIM];
extern const float MODEL_DATA_STORAGE final_norm_shift[EMB_DIM];
extern const float MODEL_DATA_STORAGE out_head_weight[VOCAB_SIZE][EMB_DIM];
"""
        return declarations

    def _generate_global_buffer_declarations(self):
        """Generates the global buffer declarations for the header file."""
        return """
// Buffers globais
extern float g_static_attn_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_norm_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_ff_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_lin1_out_buffer[4 * EMB_DIM];
extern float g_static_q_buffer[EMB_DIM];
extern float g_static_k_buffer[EMB_DIM];
extern float g_static_v_buffer[EMB_DIM];
extern float g_static_attn_scores_buffer[CONTEXT_LENGTH];
extern float g_static_attention_results_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_gpt_forward_x_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_gpt_forward_norm_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_logits_buffer[VOCAB_SIZE];
extern int g_static_tokens_buffer[CONTEXT_LENGTH]; 
extern int g_static_prompt_tokens_temp_buffer[MAX_PROMPT_TOKENS_BUFFER_SIZE];
"""

    def _generate_function_declarations(self):
        """Generates function declarations for the header file."""
        return """
// Function Declarations
void gpt_forward(float* logits_output, int* input_ids, int seq_len);
int generate_next_token(int* current_sequence_token_ids, int current_seq_len);
void generate_qna_answer(char* output_buffer, int buffer_size, const String& question_prompt, int max_new_tokens = 20);

// Tokenization Functions
char _to_lower_char(char c);
bool _is_word_char(char c);
int find_token_id(const char* word_to_find);
int tokenize_prompt(const String& prompt_str, int* out_token_ids, int max_tokens_to_fill);
const char* decode_token_str(int token_id);
"""

    def _generate_cpp_helper_functions(self):
        """Generates C++ helper functions for the .cpp file."""
        return """
// Helper functions for tokenization
char _to_lower_char(char c) { return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c; }

bool _is_word_char(char c) {
    return ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '<' || c == '>' || c == '_');
}

// Finds the ID of a token in the vocabulary (AVR and ARM supported)
int find_token_id(const char* word_to_find) {
    char buffer[MAX_TOKEN_STRING_LENGTH + 1];
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        strncpy(buffer, token_strings[i], MAX_TOKEN_STRING_LENGTH);
        buffer[MAX_TOKEN_STRING_LENGTH] = '\\0';
        if (strcmp(word_to_find, buffer) == 0) return i;
    }
    return UNK_TOKEN_ID;
}

// Tokenize a prompt into token IDs
int tokenize_prompt(const String& prompt_str, int* out_token_ids, int max_tokens_to_fill) {
    int token_count = 0;
    char current_word_buffer[MAX_TOKEN_STRING_LENGTH + 1];
    int word_buffer_idx = 0;

    for (int i = 0; i < prompt_str.length() && token_count < max_tokens_to_fill; ++i) {
        char c = prompt_str.charAt(i);
        char lower_c = _to_lower_char(c);

        if (_is_word_char(lower_c)) {
            if (word_buffer_idx < MAX_TOKEN_STRING_LENGTH) {
                current_word_buffer[word_buffer_idx++] = lower_c;
            }
        } else {
            if (word_buffer_idx > 0) {
                current_word_buffer[word_buffer_idx] = '\\0';
                out_token_ids[token_count++] = find_token_id(current_word_buffer);
                word_buffer_idx = 0;
            }
        }
    }
    if (word_buffer_idx > 0 && token_count < max_tokens_to_fill) {
        current_word_buffer[word_buffer_idx] = '\\0';
        out_token_ids[token_count++] = find_token_id(current_word_buffer);
    }
    return token_count;
}
"""

    def _generate_token_string_definitions(self):
        """Generates the token string table definition for the .cpp file."""
        decode_string_array_cpp_elements = []
        for token in self.vocab_list:
            escaped_token = str(token).replace('"', r'\"')
            decode_string_array_cpp_elements.append(f'"{escaped_token}"')
        return f"const char* const token_strings[VOCAB_SIZE] = {{ {', '.join(decode_string_array_cpp_elements)} }};\n"

    def _generate_decode_token_str_function(self):
        """Gera a função decode_token_str para o arquivo .cpp."""
        return """const char* decode_token_str(int token_id) {
    if (token_id >= 0 && token_id < VOCAB_SIZE) {
        return token_strings[token_id];
    }
    return "?";
}"""

    def _generate_cpp_math_functions(self):
        """Generates basic mathematical functions for the .cpp file."""
        return """
// Funções matemáticas básicas
void mat_vec_mul(float* out, const float* W, const float* b, const float* x, int rows_W, int cols_W) {
    for (int i = 0; i < rows_W; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < cols_W; j++) {
            out[i] += READ_MODEL_FLOAT(W[i * cols_W + j]) * x[j];
        }
        if (b != nullptr) {
            out[i] += READ_MODEL_FLOAT(b[i]);
        }
    }
}

void layer_norm(float* out, const float* x, const float* scale, const float* shift, int n, float eps = 1e-5f) {
    if (n <= 0) return;
    float mean = 0.0f; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    float var = 0.0f; for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean); var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        out[i] = READ_MODEL_FLOAT(scale[i]) * (x[i] - mean) * inv_std + READ_MODEL_FLOAT(shift[i]);
    }
}

void gelu(float* x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
    }
}

void softmax(float* x, int n) {
    if (n <= 0) return;
    float max_val = x[0]; for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    if (sum == 0.0f) sum = 1e-9f;
    for (int i = 0; i < n; i++) x[i] /= sum;
}
"""

    def _generate_cpp_attention_and_transformer_block(self):
        """Generates C++ functions for multi-head attention and transformer blocks."""
        n_layers = self.model_config['n_layers']
        n_heads = self.model_config['n_heads']

        if n_layers == 0 or n_heads == 0:
            return "" 

        return f"""
#if N_LAYERS > 0 && N_HEADS > 0
void multi_head_attention(float* out_final_seq, float* x_input_seq, int seq_len, int layer_idx) {{
    float q_vec[EMB_DIM], k_vec[EMB_DIM], v_vec[EMB_DIM];
    float attn_scores_for_head[CONTEXT_LENGTH];
    float context_vec_for_token_head[HEAD_DIM];
    float concatenated_heads_for_token[EMB_DIM];

    for (int t = 0; t < seq_len; ++t) {{
        memset(concatenated_heads_for_token, 0, EMB_DIM * sizeof(float));

        for (int h = 0; h < N_HEADS; ++h) {{
            mat_vec_mul(q_vec, (const float*)trf_att_W_query_weight[layer_idx], nullptr, &x_input_seq[t*EMB_DIM], EMB_DIM, EMB_DIM);
            
            memset(attn_scores_for_head, 0, (t + 1) * sizeof(float));
            for (int j = 0; j <= t; ++j) {{
                mat_vec_mul(k_vec, (const float*)trf_att_W_key_weight[layer_idx], nullptr, &x_input_seq[j*EMB_DIM], EMB_DIM, EMB_DIM);
                
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {{
                    score += q_vec[h * HEAD_DIM + d] * k_vec[h * HEAD_DIM + d];
                }}
                attn_scores_for_head[j] = score / sqrtf((float)HEAD_DIM);
            }}
            softmax(attn_scores_for_head, t + 1);

            memset(context_vec_for_token_head, 0, HEAD_DIM * sizeof(float));
            for (int j = 0; j <= t; ++j) {{
                mat_vec_mul(v_vec, (const float*)trf_att_W_value_weight[layer_idx], nullptr, &x_input_seq[j*EMB_DIM], EMB_DIM, EMB_DIM);
                for (int d = 0; d < HEAD_DIM; ++d) {{
                    context_vec_for_token_head[d] += attn_scores_for_head[j] * v_vec[h * HEAD_DIM + d];
                }}
            }}
            for (int d = 0; d < HEAD_DIM; ++d) {{
                concatenated_heads_for_token[h * HEAD_DIM + d] = context_vec_for_token_head[d];
            }}
        }}

        mat_vec_mul(&out_final_seq[t*EMB_DIM], (const float*)trf_att_out_proj_weight[layer_idx],
                   (const float*)trf_att_out_proj_bias[layer_idx], concatenated_heads_for_token,
                   EMB_DIM, EMB_DIM);
    }}
}}

void transformer_block(float* x_seq, int seq_len, int layer_idx) {{
    float* attn_out_seq = g_static_attn_out_buffer; 
    float* norm_out_intermediate1_seq = g_static_norm_out_buffer; 
    float* ff_out_seq = g_static_ff_out_buffer; 
    float lin1_out_vec[4 * EMB_DIM]; 
    float x_seq_copy_for_residual[CONTEXT_LENGTH * EMB_DIM];

    memcpy(x_seq_copy_for_residual, x_seq, seq_len * EMB_DIM * sizeof(float));

    for (int t = 0; t < seq_len; ++t) {{
        layer_norm(&norm_out_intermediate1_seq[t*EMB_DIM], &x_seq[t*EMB_DIM], 
                   (const float*)trf_norm1_scale[layer_idx], (const float*)trf_norm1_shift[layer_idx], EMB_DIM);
    }}
    multi_head_attention(attn_out_seq, norm_out_intermediate1_seq, seq_len, layer_idx);
    for (int i = 0; i < seq_len * EMB_DIM; i++) {{ 
        x_seq[i] = x_seq_copy_for_residual[i] + attn_out_seq[i];
    }}

    memcpy(x_seq_copy_for_residual, x_seq, seq_len * EMB_DIM * sizeof(float));

    for (int t = 0; t < seq_len; ++t) {{
        layer_norm(&norm_out_intermediate1_seq[t*EMB_DIM], &x_seq[t*EMB_DIM], 
                   (const float*)trf_norm2_scale[layer_idx], (const float*)trf_norm2_shift[layer_idx], EMB_DIM);
    }}
    for (int t = 0; t < seq_len; t++) {{ 
        mat_vec_mul(lin1_out_vec, (const float*)trf_ff_lin1_weight[layer_idx], (const float*)trf_ff_lin1_bias[layer_idx], 
                    &norm_out_intermediate1_seq[t * EMB_DIM], 4 * EMB_DIM, EMB_DIM);
        gelu(lin1_out_vec, 4 * EMB_DIM);
        mat_vec_mul(&ff_out_seq[t * EMB_DIM], (const float*)trf_ff_lin2_weight[layer_idx], (const float*)trf_ff_lin2_bias[layer_idx], 
                    lin1_out_vec, EMB_DIM, 4 * EMB_DIM);
    }}
    for (int i = 0; i < seq_len * EMB_DIM; i++) {{ 
        x_seq[i] = x_seq_copy_for_residual[i] + ff_out_seq[i];
    }}
}}
#endif
"""

    def _generate_cpp_gpt_forward(self):
        """Generates the function gpt_forward C++."""
        n_layers = self.model_config['n_layers']
        
        return f"""
void gpt_forward(float* logits_output, int* input_ids, int seq_len) {{
    if (seq_len <= 0 || VOCAB_SIZE == 0 || EMB_DIM == 0 || CONTEXT_LENGTH == 0) {{ 
        if (VOCAB_SIZE > 0) {{ for(int k=0; k < VOCAB_SIZE; ++k) logits_output[k] = 0.0f; }}
        return;
    }}

    float* x_buffer = g_static_gpt_forward_x_buffer; 
    float* norm_out_final_buffer = g_static_gpt_forward_norm_out_buffer; 

    for (int i = 0; i < seq_len; i++) {{
        int token_id = input_ids[i];
        if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = UNK_TOKEN_ID;
        if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = 0; // Fallback to 0 if UNK_TOKEN_ID is also invalid

        int pos_id = i;

        for (int j = 0; j < EMB_DIM; j++) {{
            float tok_emb_val = READ_MODEL_FLOAT(tok_emb_weight[token_id][j]);
            float pos_emb_val = READ_MODEL_FLOAT(pos_emb_weight[pos_id][j]);
            x_buffer[i * EMB_DIM + j] = tok_emb_val + pos_emb_val;
        }}
    }}

#if N_LAYERS > 0
    for (int l = 0; l < N_LAYERS; l++) {{ transformer_block(x_buffer, seq_len, l); }}
#endif

    for (int t = 0; t < seq_len; ++t) {{
        layer_norm(&norm_out_final_buffer[t*EMB_DIM], &x_buffer[t*EMB_DIM], 
                   (const float*)final_norm_scale, (const float*)final_norm_shift, EMB_DIM);
    }}
    
    int last_token_idx_in_seq = seq_len - 1;
    mat_vec_mul(logits_output, (const float*)out_head_weight, nullptr, 
                &norm_out_final_buffer[last_token_idx_in_seq * EMB_DIM], VOCAB_SIZE, EMB_DIM);
}}
"""

    def _generate_cpp_generate_next_token(self):
        """Generates the generate_next_token C++ function."""
        return """
int generate_next_token(int* current_sequence_token_ids, int current_seq_len) {
    if (VOCAB_SIZE == 0) return UNK_TOKEN_ID; 
    if (current_seq_len <= 0) return UNK_TOKEN_ID;
    
    float* logits = g_static_logits_buffer;
    gpt_forward(logits, current_sequence_token_ids, current_seq_len);
    
    int max_idx = 0;
    if (VOCAB_SIZE > 0) {
        float max_val = logits[0]; 
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
        }
    }
    return max_idx;
}
"""

    def _generate_cpp_generate_qna_answer(self):
        """Generates the generate_qna_answer C++ function."""
        return """
void generate_qna_answer(char* output_buffer, int buffer_size, const String& question_prompt, int max_new_tokens) {
    if (buffer_size <= 0 || VOCAB_SIZE == 0 || CONTEXT_LENGTH == 0 || EMB_DIM == 0) {
        if (buffer_size > 0) output_buffer[0] = '\\0'; // Ensure null termination
        return; 
    }
    memset(output_buffer, 0, buffer_size);

    String full_input_prompt_str = "";
    if (SOS_TOKEN_ID != -1) {
        full_input_prompt_str += token_strings[SOS_TOKEN_ID];
        full_input_prompt_str += " ";
    }
    full_input_prompt_str += question_prompt;
    if (SEP_TOKEN_ID != -1) {
        full_input_prompt_str += " ";
        full_input_prompt_str += token_strings[SEP_TOKEN_ID];
    }

    int* prompt_tokens_buffer = g_static_prompt_tokens_temp_buffer;
    int num_full_prompt_tokens = tokenize_prompt(full_input_prompt_str, prompt_tokens_buffer, MAX_PROMPT_TOKENS_BUFFER_SIZE);

    int* current_context_ids = g_static_tokens_buffer;
    int current_context_len = 0;
    int start_idx_for_context = 0;
    if (num_full_prompt_tokens > CONTEXT_LENGTH) {
        start_idx_for_context = num_full_prompt_tokens - CONTEXT_LENGTH;
    }
    for (int i = 0; i < CONTEXT_LENGTH && (start_idx_for_context + i) < num_full_prompt_tokens; ++i) {
        current_context_ids[current_context_len++] = prompt_tokens_buffer[start_idx_for_context + i];
    }
    
    int output_char_idx = 0;
    for (int k = 0; k < max_new_tokens; ++k) {
        if (output_char_idx >= buffer_size - (MAX_TOKEN_STRING_LENGTH + 2)) break; // +2 for space and null terminator
        if (current_context_len == 0) break;

        int next_token_id = generate_next_token(current_context_ids, current_context_len);

        if (next_token_id == EOS_TOKEN_ID || next_token_id == PAD_TOKEN_ID) break;
        if (next_token_id == SOS_TOKEN_ID || next_token_id == SEP_TOKEN_ID) continue;

        // Shift context window if it's full
        if (current_context_len >= CONTEXT_LENGTH) {
            for (int j = 0; j < CONTEXT_LENGTH - 1; j++) current_context_ids[j] = current_context_ids[j+1];
            current_context_ids[CONTEXT_LENGTH - 1] = next_token_id;
        } else {
            current_context_ids[current_context_len++] = next_token_id;
        }

        const char* decoded_token_ptr = decode_token_str(next_token_id);
        int decoded_len = strlen(decoded_token_ptr);

        if (decoded_len > 0) {
            bool add_space = (output_char_idx > 0 && output_buffer[output_char_idx - 1] != ' ');
            char first_char_of_token = decoded_token_ptr[0];
            // Don't add space before punctuation or certain special characters
            if (strchr(",.!?;:)", first_char_of_token) != nullptr) add_space = false;
            
            int space_len = add_space ? 1 : 0;
            if (output_char_idx + space_len + decoded_len < buffer_size -1) {
                if (add_space) output_buffer[output_char_idx++] = ' ';
                strncpy(&output_buffer[output_char_idx], decoded_token_ptr, decoded_len);
                output_char_idx += decoded_len;
            } else {
                break;
            }
        }
        if (output_char_idx >= buffer_size - 1) break; // Ensure space for null terminator
    }
    output_buffer[output_char_idx] = '\\0';
}
"""

    def generate_arduino_code(self, json_files, model_name):
        """
        Generates Arduino (AVR and ARM) compatible C++ code from Q&A template JSON files.
        """
        if not self._load_json_files(json_files):
            return
        if not self._validate_required_files():
            return
        if not self._process_vocabulary():
            return
        if not self._extract_config_parameters():
            return
        if not self._map_special_tokens():
            return

        trf_params_str_map = self._prepare_transformer_parameters()

        # File header .h
        header_code = f"""// Automatically generated file - {model_name}.h
#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

#include <Arduino.h>
#include <math.h>
#include <string.h>

// Architecture detection
#if defined(__AVR__)
  #include <avr/pgmspace.h>
  #define MODEL_DATA_STORAGE PROGMEM
  #define READ_MODEL_FLOAT(x) pgm_read_float_near(&x)
#else
  #define MODEL_DATA_STORAGE
  #define READ_MODEL_FLOAT(x) x
#endif
namespace {model_name} {{
{self._generate_header_defines()}

{self._generate_header_weight_declarations()}

{self._generate_global_buffer_declarations()}

{self._generate_function_declarations()}

}} // namespace {model_name}

#endif // {model_name.upper()}_MODEL_H
"""
        # File implementation .cpp
        impl_code = f"""// Automatically generated file - {model_name}_model.cpp
#include "{model_name}_model.h"

namespace {model_name} {{
// Weight definitions
const float MODEL_DATA_STORAGE tok_emb_weight[VOCAB_SIZE][EMB_DIM] = {self._CppFormatter.format_2d_array(self.params_data.get('embedding.json', {}).get('tok_emb.weight', []))};
const float MODEL_DATA_STORAGE pos_emb_weight[CONTEXT_LENGTH][EMB_DIM] = {self._CppFormatter.format_2d_array(self.params_data.get('embedding.json', {}).get('pos_emb.weight', []))};

#if N_LAYERS > 0
const float MODEL_DATA_STORAGE trf_att_W_query_weight[N_LAYERS][EMB_DIM][EMB_DIM] = {trf_params_str_map["trf_att_W_query_weight"]};
const float MODEL_DATA_STORAGE trf_att_W_key_weight[N_LAYERS][EMB_DIM][EMB_DIM] = {trf_params_str_map["trf_att_W_key_weight"]};
const float MODEL_DATA_STORAGE trf_att_W_value_weight[N_LAYERS][EMB_DIM][EMB_DIM]= {trf_params_str_map["trf_att_W_value_weight"]};
const float MODEL_DATA_STORAGE trf_att_out_proj_weight[N_LAYERS][EMB_DIM][EMB_DIM] = {trf_params_str_map["trf_att_out_proj_weight"]};
const float MODEL_DATA_STORAGE trf_att_out_proj_bias[N_LAYERS][EMB_DIM] = {trf_params_str_map["trf_att_out_proj_bias"]};
const float MODEL_DATA_STORAGE trf_ff_lin1_weight[N_LAYERS][4*EMB_DIM][EMB_DIM]= {trf_params_str_map["trf_ff_lin1_weight"]};
const float MODEL_DATA_STORAGE trf_ff_lin1_bias[N_LAYERS][4*EMB_DIM] = {trf_params_str_map["trf_ff_lin1_bias"]};
const float MODEL_DATA_STORAGE trf_ff_lin2_weight[N_LAYERS][EMB_DIM][4*EMB_DIM] = {trf_params_str_map["trf_ff_lin2_weight"]};
const float MODEL_DATA_STORAGE trf_ff_lin2_bias[N_LAYERS][EMB_DIM] = {trf_params_str_map["trf_ff_lin2_bias"]};
const float MODEL_DATA_STORAGE trf_norm1_scale[N_LAYERS][EMB_DIM] = {trf_params_str_map["trf_norm1_scale"]};
const float MODEL_DATA_STORAGE trf_norm1_shift[N_LAYERS][EMB_DIM] = {trf_params_str_map["trf_norm1_shift"]};
const float MODEL_DATA_STORAGE trf_norm2_scale[N_LAYERS][EMB_DIM] = {trf_params_str_map["trf_norm2_scale"]};
const float MODEL_DATA_STORAGE trf_norm2_shift[N_LAYERS][EMB_DIM] = {trf_params_str_map["trf_norm2_shift"]};
#endif

const float MODEL_DATA_STORAGE final_norm_scale[EMB_DIM] = {self._CppFormatter.format_array(self.params_data.get('final_norm.json', {}).get('final_norm.scale', []))};
const float MODEL_DATA_STORAGE final_norm_shift[EMB_DIM]= {self._CppFormatter.format_array(self.params_data.get('final_norm.json', {}).get('final_norm.shift', []))};
const float MODEL_DATA_STORAGE out_head_weight[VOCAB_SIZE][EMB_DIM] = {self._CppFormatter.format_2d_array(self.params_data.get('out_head.json', {}).get('out_head.weight', []))};

// Token string table
{self._generate_token_string_definitions()}

// Definição dos buffers globais
float g_static_attn_out_buffer[CONTEXT_LENGTH * EMB_DIM];
float g_static_norm_out_buffer[CONTEXT_LENGTH * EMB_DIM];
float g_static_ff_out_buffer[CONTEXT_LENGTH * EMB_DIM];
float g_static_lin1_out_buffer[4 * EMB_DIM];
float g_static_q_buffer[EMB_DIM];
float g_static_k_buffer[EMB_DIM];
float g_static_v_buffer[EMB_DIM];
float g_static_attn_scores_buffer[CONTEXT_LENGTH];
float g_static_attention_results_buffer[CONTEXT_LENGTH * EMB_DIM];
float g_static_gpt_forward_x_buffer[CONTEXT_LENGTH * EMB_DIM];
float g_static_gpt_forward_norm_out_buffer[CONTEXT_LENGTH * EMB_DIM];
float g_static_logits_buffer[VOCAB_SIZE];
int g_static_tokens_buffer[CONTEXT_LENGTH];
int g_static_prompt_tokens_temp_buffer[MAX_PROMPT_TOKENS_BUFFER_SIZE];

{self._generate_cpp_math_functions()}

{self._generate_cpp_attention_and_transformer_block()}

{self._generate_cpp_gpt_forward()}

{self._generate_cpp_generate_next_token()}

{self._generate_cpp_helper_functions()}
{self._generate_decode_token_str_function()}

{self._generate_cpp_generate_qna_answer()}
}} // namespace {model_name}
"""

        # Escreve os arquivos de saída
        with open(os.path.join(self.output_dir, f"{model_name}_model.h"), 'w', encoding='utf-8') as f:
            f.write(header_code)
        with open(os.path.join(self.output_dir, f"{model_name}_model.cpp"), 'w', encoding='utf-8') as f:
            f.write(impl_code)
        print(f"[INFO] Compatible Arduino code generated at {self.output_dir}/{model_name}_model.h and {self.output_dir}/{model_name}_model.cpp")