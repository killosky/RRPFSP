def transpose_list_of_tensors(list_of_tensors):
    max_length = max(len(sublist) for sublist in list_of_tensors)

    transposed_list = [[] for _ in range(max_length)]

    for sublist in list_of_tensors:
        for i_sublist, tensor in enumerate(sublist):
            transposed_list[i_sublist].append(tensor)

    return transposed_list
