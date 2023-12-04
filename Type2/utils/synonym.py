def generate_synonym_list_by_dict(syn_dict, word):


    if not word in syn_dict:
        return [word]
    else:
        return syn_dict[word]