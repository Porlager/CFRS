(() => {
  const body = document.body;
  const refreshSeconds = Number(body.dataset.refreshSeconds || "12");
  const cameraRefreshMs = 850;
  let latestPayload = null;

  const els = {
    days: document.getElementById("window-days"),
    search: document.getElementById("attendance-search"),
    refreshBtn: document.getElementById("refresh-btn"),
    fullscreenBtn: document.getElementById("fullscreen-btn"),
    cameraFrame: document.getElementById("camera-frame"),
    cameraStatus: document.getElementById("camera-status"),
    livePersonCount: document.getElementById("live-person-count"),
    liveAttentiveCount: document.getElementById("live-attentive-count"),
    liveInattentiveCount: document.getElementById("live-inattentive-count"),
    liveDrowsyCount: document.getElementById("live-drowsy-count"),
    liveUnknownCount: document.getElementById("live-unknown-count"),
    liveStudentsList: document.getElementById("live-students-list"),
    toast: document.getElementById("toast"),
    apiStatus: document.getElementById("api-status"),
    lastUpdated: document.getElementById("last-updated"),
    trendDateRange: document.getElementById("trend-date-range"),
    trendBars: document.getElementById("trend-bars"),
    latestCheckinName: document.getElementById("latest-checkin-name"),
    latestCheckinTime: document.getElementById("latest-checkin-time"),
    latestCheckinConfidence: document.getElementById("latest-checkin-confidence"),
    attendanceChipList: document.getElementById("attendance-chip-list"),
    distribution: document.getElementById("state-distribution"),
    attendanceBody: document.getElementById("recent-attendance-body"),
    drowsyRank: document.getElementById("drowsy-rank"),
    kpiAttendance: document.getElementById("kpi-attendance"),
    kpiAttendanceRate: document.getElementById("kpi-attendance-rate"),
    kpiTotalActive: document.getElementById("kpi-total-active"),
    kpiDrowsyStudents: document.getElementById("kpi-drowsy-students"),
    kpiBehaviorEvents: document.getElementById("kpi-behavior-events"),
    kpiUnknownEvents: document.getElementById("kpi-unknown-events"),
    refreshSec: document.getElementById("refresh-sec"),
  };

  els.refreshSec.textContent = String(refreshSeconds);

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => {
      els.toast.hidden = true;
    }, 3200);
  }

  function setApiState(ok) {
    if (!els.apiStatus) return;
    if (ok) {
      els.apiStatus.textContent = "Online";
      els.apiStatus.classList.remove("bad");
      els.apiStatus.classList.add("ok");
    } else {
      els.apiStatus.textContent = "Offline";
      els.apiStatus.classList.remove("ok");
      els.apiStatus.classList.add("bad");
    }
  }

  function setCameraStatus(text, isOk) {
    if (!els.cameraStatus) return;
    els.cameraStatus.textContent = text;
    els.cameraStatus.classList.toggle("bad", !isOk);
  }

  function formatDateTime(iso) {
    if (!iso) return "-";
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return iso;
    return dt.toLocaleString("th-TH", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function shortDay(isoDate) {
    const dt = new Date(`${isoDate}T00:00:00`);
    if (Number.isNaN(dt.getTime())) return isoDate;
    return dt.toLocaleDateString("th-TH", { month: "2-digit", day: "2-digit" });
  }

  function escapeHtml(text) {
    return String(text ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function classifyState(state) {
    const normalized = String(state || "").toLowerCase();
    if (normalized.includes("ไม่ตั้งใจ") || normalized.includes("inattentive") || normalized.includes("away")) {
      return "inattentive";
    }
    if (normalized.includes("หลับ") || normalized.includes("เหม่อ") || normalized.includes("drowsy") || normalized.includes("sleep")) {
      return "drowsy";
    }
    if (normalized.includes("ตั้งใจ") || normalized.includes("attentive")) {
      return "attentive";
    }
    return "unknown";
  }

  async function loadDashboard() {
    const days = Number(els.days.value || "7");
    const response = await fetch(`/api/reports/dashboard?days=${days}&recent_limit=12`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`dashboard_api_failed_${response.status}`);
    }
    const payload = await response.json();
    latestPayload = payload;
    render(payload);
  }

  function render(payload) {
    const data = payload || {};
    const kpis = data.kpis || {};
    const runtime = data.runtime || {};

    els.kpiAttendance.textContent = String(kpis.attendance_today ?? 0);
    els.kpiAttendanceRate.textContent = `${Number(kpis.attendance_rate_today ?? 0).toFixed(1)}%`;
    els.kpiTotalActive.textContent = String(kpis.total_active_students ?? 0);
    els.kpiDrowsyStudents.textContent = String(kpis.drowsy_students_today ?? 0);
    els.kpiBehaviorEvents.textContent = String(kpis.behavior_events_today ?? 0);
    els.kpiUnknownEvents.textContent = String(kpis.unknown_events_today ?? 0);

    if (els.livePersonCount) {
      els.livePersonCount.textContent = String(runtime.person_count ?? 0);
    }
    if (els.liveAttentiveCount) {
      els.liveAttentiveCount.textContent = String(runtime.attentive_count ?? 0);
    }
    if (els.liveInattentiveCount) {
      els.liveInattentiveCount.textContent = String(runtime.inattentive_count ?? 0);
    }
    if (els.liveDrowsyCount) {
      els.liveDrowsyCount.textContent = String(runtime.drowsy_count ?? 0);
    }
    if (els.liveUnknownCount) {
      els.liveUnknownCount.textContent = String(runtime.unknown_count ?? 0);
    }

    const currentStudents = Array.isArray(runtime.current_students) ? runtime.current_students : [];
    if (els.liveStudentsList) {
      if (currentStudents.length === 0) {
        els.liveStudentsList.innerHTML =
          `<div class="live-student-row"><div class="live-student-name">ยังไม่พบคนในห้องจาก feed ล่าสุด</div><div></div><div class="live-student-meta">รอข้อมูลจากกล้อง...</div></div>`;
      } else {
        const rows = currentStudents.map((s) => {
          const stateClass = classifyState(s.state);
          const confidenceText = s.confidence == null ? "-" : `${Number(s.confidence).toFixed(1)}%`;
          const trackLabel = s.track_id == null ? "track:-" : `track:${s.track_id}`;
          const verifyLabel = s.confirmed ? "confirmed" : "tracking";
          return `
            <div class="live-student-row">
              <div>
                <div class="live-student-name">${escapeHtml(s.name || "Unknown")}</div>
                <div class="live-student-meta">${trackLabel} | conf:${confidenceText} | ${verifyLabel}</div>
              </div>
              <div><span class="state-badge ${stateClass}">${escapeHtml(s.state || "ไม่ทราบสถานะ")}</span></div>
              <div class="live-student-meta">${escapeHtml(formatDateTime(runtime.timestamp))}</div>
            </div>
          `;
        });
        els.liveStudentsList.innerHTML = rows.join("");
      }
    }

    const latest = data.latest_checkin || null;
    els.latestCheckinName.textContent = latest ? latest.full_name : "ยังไม่มีการเช็คชื่อ";
    els.latestCheckinTime.textContent = latest ? formatDateTime(latest.first_seen_at) : "-";
    els.latestCheckinConfidence.textContent = latest
      ? `Confidence: ${latest.confidence == null ? "-" : `${Number(latest.confidence).toFixed(1)}%`}`
      : "Confidence: -";

    const todayList = data.attendance_today_list || [];
    els.attendanceChipList.innerHTML =
      todayList.length === 0
        ? `<span class="chip">ยังไม่มีรายชื่อเช็คชื่อวันนี้</span>`
        : todayList
            .slice(0, 24)
            .map(
              (r) =>
                `<span class="chip">${r.full_name} <small>${formatDateTime(r.first_seen_at)}</small></span>`
            )
            .join("");

    const trend = data.trend || [];
    if (trend.length > 0) {
      const start = trend[0].day;
      const end = trend[trend.length - 1].day;
      els.trendDateRange.textContent = `${start} → ${end}`;
    } else {
      els.trendDateRange.textContent = "-";
    }

    const maxAttendance = Math.max(1, ...trend.map((d) => Number(d.attendance || 0)));
    const maxDrowsy = Math.max(1, ...trend.map((d) => Number(d.drowsy_events || 0)));

    els.trendBars.innerHTML = trend
      .map((d) => {
        const attendancePct = Math.round((Number(d.attendance || 0) / maxAttendance) * 100);
        const drowsyPct = Math.round((Number(d.drowsy_events || 0) / maxDrowsy) * 100);
        return `
          <div class="day-col">
            <div class="day-label">${shortDay(d.day)}</div>
            <div class="bar-wrap"><div class="bar attendance" style="width:${attendancePct}%"></div></div>
            <div class="bar-wrap"><div class="bar drowsy" style="width:${drowsyPct}%"></div></div>
          </div>
        `;
      })
      .join("");

    const dist = data.state_distribution || [];
    els.distribution.innerHTML =
      dist.length === 0
        ? `<div class="state-row"><span>ยังไม่มีข้อมูลวันนี้</span><strong>0</strong></div>`
        : dist
            .map((s) => `<div class="state-row"><span>${s.state}</span><strong>${s.total}</strong></div>`)
            .join("");

    const keyword = (els.search.value || "").trim().toLowerCase();
    const recent = (data.recent_attendance || []).filter((r) => {
      if (!keyword) return true;
      return String(r.full_name || "").toLowerCase().includes(keyword);
    });

    els.attendanceBody.innerHTML =
      recent.length === 0
        ? `<tr><td colspan="3">ไม่พบข้อมูลเช็คชื่อที่ตรงกับคำค้น</td></tr>`
        : recent
            .map(
              (r) =>
                `<tr>
                  <td>${r.full_name}</td>
                  <td>${formatDateTime(r.first_seen_at)}</td>
                  <td>${r.confidence == null ? "-" : `${Number(r.confidence).toFixed(1)}%`}</td>
                </tr>`
            )
            .join("");

    const rank = data.drowsy_rank || [];
    els.drowsyRank.innerHTML =
      rank.length === 0
        ? `<li class="rank-item"><span>ไม่มีนักเรียนที่หลับ/เหม่อในช่วงนี้</span><strong>0</strong></li>`
        : rank
            .map(
              (r) =>
                `<li class="rank-item">
                  <span>${r.full_name}<br><small>${formatDateTime(r.last_seen)}</small></span>
                  <strong>${r.drowsy_count}</strong>
                </li>`
            )
            .join("");

    const now = new Date();
    els.lastUpdated.textContent = now.toLocaleTimeString("th-TH", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  async function refreshWithErrorHandling() {
    try {
      await loadDashboard();
      setApiState(true);
    } catch (err) {
      setApiState(false);
      showToast("โหลดข้อมูล dashboard ไม่สำเร็จ");
      console.error(err);
    }
  }

  function toggleFullScreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {
        showToast("ไม่สามารถเข้าโหมดเต็มจอได้");
      });
      return;
    }
    document.exitFullscreen().catch(() => {
      showToast("ไม่สามารถออกโหมดเต็มจอได้");
    });
  }

  function refreshCameraFrame() {
    if (!els.cameraFrame) return;
    const nextSrc = `/api/camera/frame?t=${Date.now()}`;
    els.cameraFrame.src = nextSrc;
  }

  els.refreshBtn.addEventListener("click", refreshWithErrorHandling);
  els.days.addEventListener("change", refreshWithErrorHandling);
  els.search.addEventListener("input", () => {
    if (latestPayload) {
      render(latestPayload);
    }
  });
  els.fullscreenBtn.addEventListener("click", toggleFullScreen);

  if (els.cameraFrame) {
    els.cameraFrame.addEventListener("load", () => {
      setCameraStatus("กล้องออนไลน์", true);
    });

    els.cameraFrame.addEventListener("error", () => {
      setCameraStatus("ยังไม่พบภาพกล้อง (ตรวจว่าเปิด backend กล้องแล้ว)", false);
    });
  }

  refreshWithErrorHandling();
  refreshCameraFrame();
  window.setInterval(refreshWithErrorHandling, refreshSeconds * 1000);
  window.setInterval(refreshCameraFrame, cameraRefreshMs);
})();
