#!/usr/bin/env sh
# Reproduce examples/responses/*.json against a running API.
#   KEY=<api key> BASE_URL=http://127.0.0.1:8000 sh examples/run_examples.sh
# Images are read from data/raw (the dataset licence does not allow redistributing them here).
set -eu
: "${KEY:?set KEY to an API key}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RAW="data/raw"
OUT="examples/responses"
mkdir -p "$OUT"

predict() {  # name, image path under data/raw
  curl -s -H "X-API-Key: $KEY" -F "file=@$RAW/$2" "$BASE_URL/v1/predict" > "$OUT/$1.json"
}

predict 1_official_normal       casting_data/casting_data/test/ok_front/cast_ok_0_1210.jpeg
predict 2_official_defective    casting_data/casting_data/test/def_front/cast_def_0_1647.jpeg
predict 3_official_missed_but_reviewed casting_data/casting_data/test/def_front/cast_def_0_1591.jpeg
predict 4_external_defective    casting_512x512/casting_512x512/def_front/cast_def_0_9435.jpeg
predict 5_external_false_alarm_reviewed casting_512x512/casting_512x512/ok_front/cast_ok_0_2819.jpeg
predict 6_external_missed_defect casting_512x512/casting_512x512/def_front/cast_def_0_335.jpeg

# Batch: the six images in one call.
curl -s -H "X-API-Key: $KEY" \
  -F "files=@$RAW/casting_data/casting_data/test/ok_front/cast_ok_0_1210.jpeg" \
  -F "files=@$RAW/casting_data/casting_data/test/def_front/cast_def_0_1647.jpeg" \
  -F "files=@$RAW/casting_512x512/casting_512x512/ok_front/cast_ok_0_2819.jpeg" \
  "$BASE_URL/v1/predict/batch" > "$OUT/7_batch.json"

# Model info.
curl -s -H "X-API-Key: $KEY" "$BASE_URL/v1/model" > "$OUT/8_model_info.json"

# Errors: no key (401), a shell script named .png (415), docs disabled in production (404).
curl -s -F "file=@$RAW/casting_data/casting_data/test/ok_front/cast_ok_0_1210.jpeg" \
  "$BASE_URL/v1/predict" > "$OUT/9_error_no_api_key.json"
FAKE="$OUT/.not_an_image.png"
printf '#!/bin/sh\necho pwned\n' > "$FAKE"
curl -s -H "X-API-Key: $KEY" -F "file=@$FAKE;type=image/png" \
  "$BASE_URL/v1/predict" > "$OUT/10_error_fake_png.json"
rm -f "$FAKE"
curl -s "$BASE_URL/docs" > "$OUT/11_error_docs_disabled.json"

# Response headers of one prediction (security headers, request id).
curl -s -D - -o /dev/null -H "X-API-Key: $KEY" \
  -F "file=@$RAW/casting_data/casting_data/test/ok_front/cast_ok_0_1210.jpeg" \
  "$BASE_URL/v1/predict" | tr -d '\r' | grep -v -i '^date:' > "$OUT/12_response_headers.txt"
echo "wrote $(ls "$OUT" | wc -l) files to $OUT"
