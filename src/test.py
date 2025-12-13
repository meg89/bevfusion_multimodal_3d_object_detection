k = "some_key"
v = "1.234567" # The value is a string, which causes the error

# Convert v to float inside the f-string
print(f"{k}: {float(v):.4f}")
