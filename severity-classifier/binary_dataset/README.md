Merged binary dataset for damaged/undamaged.

Structure follows torchvision ImageFolder:

- train/
  - undamaged/
  - damaged/
- valid/
  - undamaged/
  - damaged/
- test/ (optional)
  - undamaged/
  - damaged/

Generate with: `bash severity-classifier/tools/merge_to_binary.sh`

