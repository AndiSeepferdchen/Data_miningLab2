import pandas as pd
import json
import os
os.chdir(os.path.dirname(__file__))

id_df = pd.read_csv("./data_identification.csv")
emotion_df = pd.read_csv("./emotion.csv")
posts = []

with open("final_posts.json", "r", encoding="utf-8") as f:
    data = json.load(f)
records = []
for item in data:
    post = item.get("root", {}).get("_source", {}).get("post", {})
    if post:
        records.append({
            "id": post.get("post_id"),
            "texts": post.get("text"),
            "hashtags": post.get("hashtags")
        })


# Convert to DataFrame
posts_df = pd.DataFrame(records, columns=["id", "texts", "hashtags"])
print(posts_df)

posts_df["texts"] = posts_df.apply(lambda row: row["texts"] + " " + " ".join(row["hashtags"]) if row["hashtags"] else row["texts"], axis=1)

# Drop the hashtags column
posts_df = posts_df.drop(columns=["hashtags"])

print(posts_df)

df_combined = pd.merge(posts_df, emotion_df, on="id", how="left")
#df_combined = df_combined.dropna(subset=["emotion"])


df_combined = pd.merge(df_combined, id_df, on="id", how="left")
df_combined = df_combined.dropna(subset=["split"])
print(df_combined)


invalid_splits = df_combined[~df_combined["split"].isin(["train", "test"])]

# Check if any exist
if not invalid_splits.empty:
    print("Invalid split values found:")
    print(invalid_splits)
else:
    print("All split values are valid.")

df_train = df_combined[df_combined["split"] == "train"].reset_index(drop=True)
print(df_train)
# Test DataFrame
df_test = df_combined[df_combined["split"] == "test"].reset_index(drop=True)
print(df_test)
df_train = df_train.drop(columns=["split"])
df_test = df_test.drop(columns=["split"])
# Save train DataFrame
df_train.to_csv("train.csv", index=False)

# Save test DataFrame
df_test.to_csv("test.csv", index=False)