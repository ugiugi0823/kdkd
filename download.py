from huggingface_hub import snapshot_download
import shutil

# 모델 ID와 저장 경로 지정
model_id = "Qwen/Qwen3.5-9B-Base"
save_dir = "/PROJECT/0325120095_A/BASE/rex/LLM/models/input/Qwen/Qwen3.5-9B-Base"





# huggingface_hub의 snapshot_download를 사용해 전체 파일 다운로드
cache_dir = snapshot_download(repo_id=model_id)

# 다운로드된 캐시를 지정한 디렉토리로 복사
shutil.copytree(cache_dir, save_dir, dirs_exist_ok=True)

print(f"✅ 전체 모델 파일이 '{save_dir}' 경로에 저장되었습니다.")

