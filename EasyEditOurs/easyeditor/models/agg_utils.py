from ..util.generate import generate_fast
from .rome import repr_tools

CONTEXT_TEMPLATES_CACHE = None

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_module(model, name):
    parts = name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def get_context_templates_rome(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_context_templates_memit(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


"""
class AggDataset:
    def __init__(self, subjects: List[str], templates: List[str], tokenizer, batch_size: int, subtoken: bool = False):
        self.subjects = subjects
        self.templates = templates
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # self.subtoken = subtoken
        
        words = [word for word in subjects for _ in templates]
        stack_templates = [temp for word in subjects for temp in templates]
        
        contexts = [stack_templates[i].format(words[i]) for i in range(len(words))]
        idxs = repr_tools.get_words_idxs_in_templates(self.tok, stack_templates, words, subtoken)
        
        self.ctx_idx_tup = [(i, contexts[i], idxs[i]) for i in range(len(contexts))]
        # random.shuffle(self.ctx_idx_tup)
        
        self.current_index = 0

    # def get_next_batch(self) -> Tuple[List[Tuple[int, str, List[int]]], int]:
    #     if self.current_index + self.batch_size > len(self.ctx_idx_tup):
    #         # Reshuffle the list if we've reached the end
    #         random.shuffle(self.ctx_idx_tup)
    #         batch = self.ctx_idx_tup[:self.batch_size]
    #         self.current_index = self.batch_size
    #     else:
    #         batch = self.ctx_idx_tup[self.current_index:self.current_index + self.batch_size]
    #         self.current_index += self.batch_size
        
    #     return batch, self.current_index

    def __len__(self) -> int:
        return len(self.ctx_idx_tup)

    def __getitem__(self, idx: int) -> Tuple[int, str, List[int]]:
        return self.ctx_idx_tup[idx]

def collate_fn(batch: List[Tuple[int, str, List[int]]]) -> Dict[str, torch.Tensor]:
    ids, contexts, word_idxs = zip(*batch)
    
    # Tokenize contexts
    # tokenized = tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
                
    batch_id = [item[1] for item in batch]
    batch_contexts = [item[1] for item in batch]
    batch_idxs = [item[2] for item in batch]
    contexts_tok = self.tok(
        batch_contexts,
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{self.hparams.device}")
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "word_idxs": torch.tensor(word_idxs),
        "ids": torch.tensor(ids)
    }
"""