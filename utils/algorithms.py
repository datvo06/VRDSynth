class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]


    def get_group(self, parent):
        out = []
        for i in range(len(self.parent)):
            if self.parent[i] == parent:
                out.append(i)
        return out


    def union(self, x, y):
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        if self.size[px] > self.size[py]:
            px, py = py, px

        self.parent[px] = py
        self.size[py] += self.size[px]

        return True

    def groups(self):
        ans = [[] for _ in range(len(self.parent))]
        for i in range(len(self.parent)):
            ans[self.find(i)].append(i)
        return list(filter(lambda x: len(x) > 0, ans))
