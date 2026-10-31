#!/usr/bin/env bash
# 宿主机调用；仅新建一次性搜索容器，数据库/源码只读。不需要Docker-in-Docker。
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo '用法：bash tools/search_msa_docker.sh FASTA DB_DIR NEW_OUT [--msa-threads 16 ...]' >&2
  echo '环境：MSA_IMAGE=localcolabfold:latest（本机），MSA_DEVICE=cpu|gpu，MSA_GPUS=0' >&2
  exit 2
fi
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
query_path=$(realpath -e -- "$1")
database_path=$(realpath -e -- "$2")
requested_out=$3
shift 3
if [[ ! -f "$query_path" || ! -d "$database_path" || -e "$requested_out" || -L "$requested_out" ]]; then
  echo '输入路径无效或输出已存在；必须提供新的目录。' >&2
  exit 2
fi
output_path=$(realpath -m -- "$requested_out")
if [[ "$output_path/" == "$database_path/"* ]]; then
  echo '输出不能位于数据库中。' >&2
  exit 2
fi
# 不允许附加参数覆盖隔离挂载或输出协议。
for argument in "$@"; do
  case "$argument" in
    --out|--out=*|--fasta|--fasta=*|--msa-local-db|--msa-local-db=*|--msa-device|--msa-device=*)
      echo "不能覆盖内部路径/设备参数：$argument" >&2; exit 2 ;;
  esac
done
image=${MSA_IMAGE:-ghcr.io/sokrypton/colabfold:1.6.3-cuda12}
device=${MSA_DEVICE:-cpu}
gpu_args=()
case "$device" in
  cpu) ;;
  gpu) gpu_args=(--gpus "\"device=${MSA_GPUS:-0}\"") ;;
  *) echo 'MSA_DEVICE须为cpu或gpu' >&2; exit 2 ;;
esac
# 不自动pull；用户先明确准备镜像，记录实际image ID。
image_id=$(docker image inspect --format '{{.Id}}' "$image")
mkdir -p -- "$(dirname -- "$output_path")"
mkdir -- "$output_path"
printf '%s\n' "$image_id" > "$output_path/docker_image_id.txt"
echo "[本地搜索容器] image=$image_id device=$device 输出=$output_path/msa"
docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=1g \
  --user "$(id -u):$(id -g)" "${gpu_args[@]}" \
  --mount "type=bind,src=$release_root,dst=/release,readonly" \
  --mount "type=bind,src=$query_path,dst=/query.fasta,readonly" \
  --mount "type=bind,src=$database_path,dst=/db,readonly" \
  --mount "type=bind,src=$output_path,dst=/work" \
  --workdir /work --entrypoint /bin/bash "$image_id" -lc \
  'cd /release && exec python3 -B -u -m oridynamic.local_msa --fasta /query.fasta --msa-local-db /db --out /work/msa "$@"' \
  bash --msa-device "$device" "$@"
if [[ ! -s "$output_path/msa/unpaired.a3m" ]]; then
  echo '未产生已验收A3M；不能启动后续推理。预检请直接使用oridynamic msa --check。' >&2
  exit 1
fi
echo "[可用于推理] --msa-a3m $output_path/msa/unpaired.a3m"
