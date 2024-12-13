from collections import OrderedDict


def merge(ids1, pair1, idx1):
    newids1 = []
    i = 0
    while i < len(ids1):
        if i < len(ids1)-1 and ids1[i] == pair1[0] and ids1[i+1] == pair1[1]:
            newids1.append(idx1)
            i+=2
        else:
            newids1.append(ids1[i])
            i+=1
    return newids1

def BytePairEncoding(vocab, num_merges):
    def getStats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    ids = list(vocab)
    merges=OrderedDict()

    for i in range(num_merges):
        stats = getStats(ids)
        pair = max(stats, key=stats.get)
        idx = 68+i
        print("Merging ", pair," into ", idx)
        ids= merge(ids, pair, idx)
        merges[pair] = idx
    
    print(merges)

    return ids, merges