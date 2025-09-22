def generate_binary_strings(n, k):
    result = []

    def backtrack(current):
        if len(current) == n:
            # kiểm tra có chứa k bit 1 liên tiếp
            if "1" * k in current:
                result.append(current)
            return
        # Thêm '0'
        backtrack(current + "0")
        # Thêm '1'
        backtrack(current + "1")

    backtrack("")
    return result


# Ví dụ:
n = 5
k = 3
strings = generate_binary_strings(n, k)
print(f"Tất cả xâu nhị phân độ dài {n} có chứa {k} bit 1 liền nhau:")
for s in strings:
    print(s)
