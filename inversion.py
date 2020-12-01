def inversion(order: list, c:float,c_accent:float):
    index_c = order.index(c)
    index_c_accent = order.index(c_accent)

    if index_c < index_c_accent:
        part1 = order[:index_c+1]
        part2 = order[index_c+1:index_c_accent+1]
        part2.reverse()
        part3 = order[index_c_accent+1:]
        order[:] = part1 + part2 + part3 # If you assign something to elements of that list, this will change the original list




    elif index_c > index_c_accent:

        part1 = order[:index_c_accent + 1]
        part2 = order[index_c_accent + 1 : index_c + 1]
        part3 = order[index_c + 1 :]

        part_reversed = part3 + part1
        part_reversed.reverse()
        order[:] = part2 + part_reversed # If you assign something to elements of that list, this will change the original list


    else:
        raise Exception("Should have been accepted by the previous two statements - inversion")