import threading

_engine_instance = None
_engine_lock = threading.Lock()

def get_grounding_engine():
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            print(">>> [Zelyr] Bootstrapping resident ZelyrGroundingEngine into CUDA memory...")
            from zelyr.core.vision_engine.zelyr_engine import ZelyrGroundingEngine
            _engine_instance = ZelyrGroundingEngine()
            ZelyrGroundingEngine._instance = _engine_instance
    return _engine_instance


def evict_grounding_engine():
    global _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            print(">>> [Zelyr] Evicting resident ZelyrGroundingEngine from CUDA memory...")
            try:
                if hasattr(_engine_instance, "unload"):
                    _engine_instance.unload()
            except Exception as e:
                print(f"Warning during vision engine eviction: {e}")
            _engine_instance = None
            try:
                from zelyr.core.vision_engine.zelyr_engine import ZelyrGroundingEngine
                ZelyrGroundingEngine._instance = None
            except Exception:
                pass
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

