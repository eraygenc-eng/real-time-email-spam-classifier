import pandas as pd

# Open the ARFF file
with open("Data/norm_spam.arff", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line after @data
data_start = None
for i, line in enumerate(lines):
    if line.strip().lower() == "@data":
        data_start = i + 1
        break

if data_start is None:
    raise ValueError("@data section not found")

rows = []

# Read actual data lines
for line in lines[data_start:]:
    line = line.strip()

    if not line:
        continue

    # Labels are attached at the end of each line
    if line.endswith("norm"):
        label = "norm"
        text = line[:-4].strip()
    elif line.endswith("spam"):
        label = "spam"
        text = line[:-4].strip()
    else:
        continue

    # Remove extra quotes around the text
    text = text.strip("'").strip('"')
    rows.append([text, label])

# Create dataframe and save it as CSV
df = pd.DataFrame(rows, columns=["text", "label"])
df.to_csv("norm_spam.csv", index=False, encoding="utf-8")

print(df.head())
print()
print(df["label"].value_counts())
print()
print("CSV file created.")