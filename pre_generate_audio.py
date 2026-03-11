import os
import json
import asyncio
import hashlib
import edge_tts

VOICE = "vi-VN-HoaiMyNeural"
OUT_DIR = "assets_audio"   # audio sinh ra để runtime dùng

def safe_name(text: str) -> str:
    return "".join(c for c in text if c.isalnum() or c in " _-").strip().replace(" ", "_")

def out_path(cid: int, text: str) -> str:
    # đặt tên theo cid để runtime tìm nhanh
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return os.path.join(OUT_DIR, f"{cid:02d}_{safe_name(text)}_{h}.mp3")

async def synth_one(cid: int, text: str):
    p = out_path(cid, text)
    if os.path.exists(p):
        return
    comm = edge_tts.Communicate(text=text, voice=VOICE)
    await comm.save(p)

async def main():
    if not os.path.exists("phrases.json"):
        raise RuntimeError("Không thấy phrases.json. Hãy chạy make_phrases.py trước.")

    os.makedirs(OUT_DIR, exist_ok=True)

    with open("phrases.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # synth tuần tự để ổn định (ít lỗi mạng). Nếu muốn nhanh hơn mình sẽ cho chạy song song có giới hạn.
    for k in sorted(mapping.keys(), key=lambda x: int(x)):
        cid = int(k)
        text = mapping[k]
        await synth_one(cid, text)
        print("Generated:", cid, text)

    print("Done. Audio in:", OUT_DIR)

if __name__ == "__main__":
    asyncio.run(main())
