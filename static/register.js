(() => {
  const els = {
    form: document.getElementById("register-form"),
    studentCode: document.getElementById("student-code"),
    fullName: document.getElementById("full-name"),
    photoFile: document.getElementById("photo-file"),
    submitBtn: document.getElementById("submit-btn"),
    clearBtn: document.getElementById("clear-btn"),
    webcam: document.getElementById("webcam"),
    canvas: document.getElementById("snapshot"),
    preview: document.getElementById("snapshot-preview"),
    startCamera: document.getElementById("start-camera"),
    captureBtn: document.getElementById("capture-btn"),
    stopCamera: document.getElementById("stop-camera"),
    cameraState: document.getElementById("camera-state"),
    resultLog: document.getElementById("result-log"),
  };

  let stream = null;
  let capturedBlob = null;

  function setLog(text) {
    els.resultLog.textContent = text;
  }

  function setCameraState(text) {
    els.cameraState.textContent = text;
  }

  async function startCamera() {
    if (stream) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      els.webcam.srcObject = stream;
      setCameraState("กล้องพร้อมใช้งาน");
    } catch (err) {
      setCameraState("เปิดกล้องไม่สำเร็จ");
      setLog(`เปิดกล้องไม่สำเร็จ: ${err.message || err}`);
    }
  }

  function stopCamera() {
    if (!stream) return;
    for (const track of stream.getTracks()) {
      track.stop();
    }
    stream = null;
    els.webcam.srcObject = null;
    setCameraState("ปิดกล้องแล้ว");
  }

  async function captureSnapshot() {
    if (!stream) {
      setLog("กรุณาเปิดกล้องก่อนกด Capture");
      return;
    }

    const videoWidth = els.webcam.videoWidth || 1280;
    const videoHeight = els.webcam.videoHeight || 720;
    els.canvas.width = videoWidth;
    els.canvas.height = videoHeight;

    const ctx = els.canvas.getContext("2d");
    ctx.drawImage(els.webcam, 0, 0, videoWidth, videoHeight);

    capturedBlob = await new Promise((resolve) => {
      els.canvas.toBlob(resolve, "image/jpeg", 0.92);
    });

    if (!capturedBlob) {
      setLog("จับภาพไม่สำเร็จ กรุณาลองใหม่");
      return;
    }

    const previewUrl = URL.createObjectURL(capturedBlob);
    els.preview.src = previewUrl;
    els.preview.hidden = false;
    setLog("จับภาพจากกล้องแล้ว สามารถกดลงทะเบียนได้เลย");
  }

  async function submitRegistration(event) {
    event.preventDefault();

    const studentCode = (els.studentCode.value || "").trim();
    const name = (els.fullName.value || "").trim();
    if (!studentCode) {
      setLog("กรุณากรอกเลขนักศึกษาก่อนลงทะเบียน");
      return;
    }
    if (!name) {
      setLog("กรุณากรอกชื่อก่อนลงทะเบียน");
      return;
    }

    const formData = new FormData();
    formData.append("student_code", studentCode);
    formData.append("full_name", name);

    const file = els.photoFile.files && els.photoFile.files[0] ? els.photoFile.files[0] : null;
    if (file) {
      formData.append("photo", file, file.name);
    } else if (capturedBlob) {
      formData.append("photo", capturedBlob, `${name.replace(/\s+/g, "_")}_capture.jpg`);
    } else {
      setLog("กรุณาอัปโหลดรูป หรือเปิดกล้องแล้ว Capture ก่อนลงทะเบียน");
      return;
    }

    els.submitBtn.disabled = true;
    try {
      const resp = await fetch("/api/register-face", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `request_failed_${resp.status}`);
      }

      setLog(
        [
          "ลงทะเบียนสำเร็จ",
          `รหัสนักศึกษา: ${data.student_code}`,
          `ชื่อ: ${data.full_name}`,
          `ไฟล์: ${data.saved_file}`,
          "หาก backend กล้องกำลังทำงานอยู่ ให้รีสตาร์ตเพื่อโหลดใบหน้าใหม่",
        ].join("\n")
      );
    } catch (err) {
      setLog(`ลงทะเบียนไม่สำเร็จ: ${err.message || err}`);
    } finally {
      els.submitBtn.disabled = false;
    }
  }

  function clearForm() {
    els.form.reset();
    capturedBlob = null;
    els.preview.hidden = true;
    els.preview.removeAttribute("src");
    setLog("ล้างค่าเรียบร้อย");
  }

  els.form.addEventListener("submit", submitRegistration);
  els.clearBtn.addEventListener("click", clearForm);
  els.startCamera.addEventListener("click", startCamera);
  els.captureBtn.addEventListener("click", captureSnapshot);
  els.stopCamera.addEventListener("click", stopCamera);

  window.addEventListener("beforeunload", stopCamera);
})();
