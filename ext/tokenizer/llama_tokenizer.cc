/*
 * Copyright (c) 2022 TENCENT CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "llama_tokenizer.h"

#include <queue>

#include "sentencepiece_processor.h"
namespace ft_ext {

//
// tokenizer
//

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score>      id_to_token;
};

static size_t utf8_len(char src)
{
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t      highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index       prev;
    index       next;
    const char* text;
    size_t      n;
};

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram& l, llama_sp_bigram& r)
        {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue         = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float                  score;
    size_t                 size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_tokenizer {
    llama_tokenizer(const llama_vocab& vocab): vocab_(vocab) {}

    void tokenize(const std::string& text, std::vector<llama_vocab::id>& output)
    {
        // split string into utf8 chars
        int    index = 0;
        size_t offs  = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t          char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text                 = text.c_str() + offs;
            sym.n                    = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto& left_sym  = symbols_[bigram.left];
            auto& right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            // printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto& symbol = symbols_[i];
            auto  token  = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int)symbol.n; ++j) {
                    llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            }
            else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right)
    {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text  = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto              token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto& tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size  = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab&           vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue       work_queue_;
};

LLamaTokenizer::LLamaTokenizer()
{
    processor_ = std::make_shared<sentencepiece::SentencePieceProcessor>();
}

int LLamaTokenizer::Load(const std::string& model, const std::string& added_tokens)
{

    auto status = processor_->Load(model);

    return 0;
}

std::vector<int> LLamaTokenizer::Encode(const std::string& text, bool add_bos)
{
    std::vector<int> output;

    processor_->Encode(text, &output);
    if (add_bos) {
        output.insert(output.begin(), processor_->bos_id());
    }

    // if (text.size() == 0) {
    //     return output;
    // }

    // llama_tokenizer tokenizer(*vocab_);
    // tokenizer.tokenize(text, output);
    return output;
}

std::string LLamaTokenizer::Decode(const std::vector<int>& ids)
{
    std::string s;
    processor_->Decode(ids, &s);
    return s;
}

}  // namespace ft_ext