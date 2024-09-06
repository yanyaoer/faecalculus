Rag based code tools
---

## setup
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## run
```python3
./faecalculus.py -h

usage: Codebase-QA [-h] [-p REPO_PATH] [-l LANGUAGE] [-i INDEX_NAME] [-m MODE]
                   [--debug DEBUG] [-q QUESTION]

How does the program work

options:
  -h, --help            show this help message and exit
  -p REPO_PATH, --repo_path REPO_PATH
  -l LANGUAGE, --language LANGUAGE
  -i INDEX_NAME, --index_name INDEX_NAME
  -m MODE, --mode MODE
  --debug DEBUG
  -q QUESTION, --question QUESTION


# build_index
./faecalculus.py -m index

# ask question
./faecalculus.py -q "how to seek audio"
```

```markdown
The provided code snippet refers to the rodio library in Rust for audio playback. It includes a test case named `seek_does_not_break_channel_order` that demonstrates how seeking in an audio file does not break the channel order.

**Understanding the Code:**

The test case iterates through various offsets and seeks to a specific beep range within the audio file. It verifies that the audio plays correctly on both channels (left and right) by checking if the left channel is silent and the right channel is not silent.

**Seek Behavior:**

The seeking operation in rodio preserves the channel order. This means that seeking to a specific point in the audio file will not change the order of the left and right channels.

**Key Points:**

- The test case uses a stereo beep file.
- It seeks to different offsets within the beep range.
- It verifies that the left channel is silent and the right channel is not silent after seeking.
- Seeking does not break the channel order.

**Conclusion:**

The `seek_does_not_break_channel_order` test case demonstrates that seeking in an audio file using the rodio library preserves the channel order, ensuring that the left and right channels are played correctly even after seeking to different points in the audio.
```
