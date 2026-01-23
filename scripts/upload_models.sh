#!/usr/bin/env bash

langs=(ibo fil lit swh zsm ron ind) 


for lang in "${langs[@]}"; do
    # upload ft models
	hf repo create MultilingualUnigramLM/ft-olmo3-7b-fineweb10M-${lang}
    hf upload MultilingualUnigramLM/ft-olmo3-7b-fineweb10M-${lang} /home/USER/unimixlm/finetuned_model/ft_${lang}_10M --repo-type model

    # upload ft-las models
	hf repo create MultilingualUnigramLM/ft-las-olmo3-7b-fineweb10M-${lang}
    hf upload MultilingualUnigramLM/ft-las-olmo3-7b-fineweb10M-${lang} /home/USER/unimixlm/finetuned_model/ft-las_${lang}_10M --repo-type model
done

date