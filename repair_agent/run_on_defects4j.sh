#!/bin/bash
export PATH=$PATH:$(pwd)/defects4j/framework/bin
cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
for LANG in en_AU.UTF-8 en_GB.UTF-8 C.UTF-8 C; do
  if locale -a 2>/dev/null | grep -q "$LANG"; then
    export LANG
    break
  fi
done
export LC_COLLATE=C

EXPERIMENT_DIR=$(python3 experimental_setups/increment_experiment.py)
echo "Creating experiment folder: $EXPERIMENT_DIR"
python3 construct_commands_descriptions.py

input="$1"
experiment_file="$2"
# $3 is an optional model shortcut. When omitted, SMART_LLM / FAST_LLM /
# STATIC_LLM env vars (or per-role flags below) control model selection.
model="$3"

dos2unix "$input"  # Convert file to Unix line endings (if needed)

ai_settings_path="experimental_setups/$EXPERIMENT_DIR/ai_settings.yaml"

model_args=()
if [ -n "$model" ]; then
    model_args+=(--model "$model")
fi

while IFS= read -r line || [ -n "$line" ]
do
    tuple=($line)
    echo ${tuple[0]}, ${tuple[1]}
    python3 prepare_ai_settings.py "${tuple[0]}" "${tuple[1]}" "$ai_settings_path"
    python3 checkout_py.py "${tuple[0]}" "${tuple[1]}"
    ./run.sh --ai-settings "$ai_settings_path" "${model_args[@]}" -c -l 40 -m json_file --experiment-file "$experiment_file" --experiment-dir "$EXPERIMENT_DIR"
done < "$input"
