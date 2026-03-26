try:
    import mediapipe as mp
    print("MediaPipe imported. Content:")
    print(dir(mp))
    try:
        print("Solutions:", dir(mp.solutions))
    except Exception as e:
        print("Error accessing solutions:", e)
    try:
        from mediapipe.python import solutions
        print("Explicit solutions import worked!")
    except Exception as e:
        print("Error importing mediapipe.python.solutions:", e)
except Exception as e:
    print("Failed to import mediapipe:", e)
