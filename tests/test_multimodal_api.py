from pathlib import Path

from fastapi.testclient import TestClient


def test_multimodal_index_search_and_rag_context(monkeypatch, tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "meeting.txt"
    source.write_text(
        "The audio transcript says the team discussed budget, pricing, "
        "remote control design, and product features."
    )

    monkeypatch.setenv("VALONY_UPLOADS_DIR", str(uploads))

    import app.main as main
    from app.multimodal import SQLiteVectorStore

    monkeypatch.setattr(main, "_MM_STORE", SQLiteVectorStore(tmp_path / "mm_api.db"))
    client = TestClient(main.app)

    index_res = client.post(
        "/v1/multimodal/index",
        json={
            "paths": [str(source)],
            "collection": "course",
            "source_type": "audio",
            "chunk_target_chars": 240,
            "chunk_overlap_chars": 20,
            "embedding_dim": 64,
        },
    )
    assert index_res.status_code == 200, index_res.text
    assert index_res.json()["chunks_indexed"] == 1
    assert index_res.json()["stats"]["by_modality"]["audio"] == 1

    search_res = client.post(
        "/v1/multimodal/search",
        json={
            "query": "remote control product features",
            "collection": "course",
            "top_k": 3,
            "embedding_dim": 64,
        },
    )
    assert search_res.status_code == 200, search_res.text
    body = search_res.json()
    assert body["results"]
    assert body["results"][0]["source_type"] == "audio"

    rag_res = client.post(
        "/v1/multimodal/rag",
        json={
            "query": "What product topics were discussed?",
            "collection": "course",
            "top_k": 3,
            "embedding_dim": 64,
            "generate": False,
        },
    )
    assert rag_res.status_code == 200, rag_res.text
    rag = rag_res.json()
    assert rag["sources"]
    assert "modality=audio" in rag["context"]
