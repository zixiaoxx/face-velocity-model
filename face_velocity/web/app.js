const $ = (id) => document.getElementById(id);
let session = null,
  snapshot = null,
  running = false,
  busy = false,
  cursor = 0;
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(2) + "%");
const num = (x) => (x == null ? "—" : Number(x).toFixed(3));
const safe = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const image = (e) =>
  `/sessions/${session}/images/${encodeURIComponent(e.event_id)}`;
async function api(url, options = {}) {
  let r = await fetch(url, options);
  let d = await r.json();
  if (!r.ok) throw Error(d.detail || r.statusText);
  return d;
}
const displayTime = (value) =>
  new Date(value).toLocaleString("en-GB", {
    timeZone: "UTC",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
function controls() {
  for (const id of ["source", "seed", "target", "submit-upload", "resume"])
    $(id).disabled = busy || running;
  $("start").disabled = busy || running;
  const hasLaterEvent = snapshot && cursor < snapshot.events.length - 1;
  $("previous").disabled =
    busy || running || !snapshot?.events.length || cursor === 0;
  $("next").disabled =
    !session ||
    busy ||
    running ||
    ($("source").value === "upload" && !hasLaterEvent);
  $("run").disabled =
    !session || busy || running || $("source").value === "upload";
  $("stop").disabled = !running;
}
function failure(e) {
  $("error").textContent = e.message;
  running = false;
  busy = false;
  controls();
}
function clearReplay() {
  snapshot = { events: [], evaluation: null };
  render();
  snapshot = null;
  $("progress").textContent = "Ready for the first submission";
  $("running-metrics").textContent = "No scored queries yet.";
  $("gallery-title").textContent = "Dynamic gallery";
  $("gallery-description").textContent =
    "Each event keeps its assigned identity. Only references take part in later searches.";
}
async function configureSource() {
  session = null;
  running = false;
  clearReplay();
  controls();
  $("error").textContent = "";
  $("upload").hidden = $("source").value !== "upload";
  $("mode").textContent =
    $("source").value === "upload"
      ? "Live image extraction with SCRFD + CASIA. Identity and reference decisions use the frozen policy."
      : "Sequential replay uses evaluated CASIA embeddings. Matching and storage run now.";
  if ($("source").value === "upload") return;
  try {
    let d = await api("/evidence/" + $("source").value);
    $("limitations").textContent = d.limitations;
    $("target").max = d.images;
  } catch (e) {
    failure(e);
  }
}
async function refresh(follow = true) {
  snapshot = await api("/sessions/" + session);
  if (follow) cursor = Math.max(0, snapshot.events.length - 1);
  render();
}
function render() {
  if (!snapshot) return;
  let events = snapshot.events;
  controls();
  $("progress").textContent =
    `${events.length.toLocaleString()} submissions processed`;
  $("scrub").hidden = !events.length;
  $("cursor").max = Math.max(0, events.length - 1);
  $("cursor").value = cursor;
  $("cursor-label").textContent = `${cursor + 1} / ${events.length}`;
  let metrics = snapshot.evaluation;
  $("running-metrics").textContent = metrics
    ? `${metrics.scored.toLocaleString()} scored queries · count accuracy ${pct(metrics.count_accuracy)} · mean Extra + Missed ${num(metrics.mean_cost)} · ${metrics.attempted - metrics.scored} unresolved`
    : "Ground truth is not available for uploaded images.";
  if (!events.length) {
    $("incoming").innerHTML = "No submissions yet.";
    $("match").innerHTML = "The gallery is empty.";
    $("count").textContent = "—";
    $("gallery").innerHTML = "";
    $("input-meta").innerHTML = "";
    $("decision").innerHTML = "";
    $("window").textContent = "";
    $("outcome").textContent = "";
    return;
  }
  let e = events[cursor],
    past = events.slice(0, cursor + 1),
    truth = $("truth").checked,
    annotation = metrics?.events[e.event_id];
  let winner = events.find((r) => r.event_id === e.winning_reference);
  $("incoming").className = "";
  $("incoming").innerHTML =
    `<img class="queryphoto" src="${image(e)}" alt="Current submission">`;
  $("input-meta").innerHTML =
    `<b>${safe(e.event_id)}</b><p class="small">${safe(displayTime(e.event_time))} UTC<br>${e.extraction.status === "valid_embedding" ? "Face encoded · 512 features" : "No usable target"}</p>`;
  $("match").className = "";
  $("match").innerHTML = winner
    ? `<div class="pair"><img class="refphoto" src="${image(winner)}" alt="Highest-scoring reference"><div><span class="small">Highest similarity</span><div class="score" title="${safe(e.top_score)}">${e.top_score.toFixed(4)}</div><span class="small">${safe(winner.event_id)}<br>Identity ${winner.assigned_identity}</span></div></div>`
    : "<p>No prior reference match.</p>";
  $("decision").innerHTML =
    `<div class="action">${e.decision === "enroll" ? "Enroll new identity" : e.decision === "accept" ? "Accept as Identity " + e.assigned_identity : "Unresolved input"}</div><p class="decisiontext">${e.reference_added ? "Store as a reference" : e.assigned_identity ? "Record event; keep references unchanged" : "No identity or reference update"}</p><p class="small">${e.reference_count_before} earlier references searched</p>`;
  $("count").textContent = e.count ?? "—";
  $("window").textContent =
    `(${displayTime(e.window_start)} → ${displayTime(e.window_end)}] UTC`;
  $("outcome").innerHTML =
    truth && annotation
      ? `<b>Evaluator true count: ${annotation.true_count}</b><p class="warn">${annotation.unresolved ? "Unresolved" : `Extra ${annotation.extra.length} · Missed ${annotation.missed.length}`}</p>`
      : "";
  let filter = $("gallery-filter").value,
    shown = past.filter(
      (r) =>
        filter === "all" ||
        (filter === "references"
          ? r.reference_added
          : r.assigned_identity === e.assigned_identity &&
            e.assigned_identity != null),
    );
  $("gallery-title").textContent =
    filter === "identity" && e.assigned_identity
      ? `Gallery after this event · Identity ${e.assigned_identity}`
      : "Gallery after this event";
  // Group by the assigned identity, never by evaluator labels.
  const identityGroups = new Map();
  for (const event of shown) {
    const identity = event.assigned_identity;
    if (!identityGroups.has(identity)) identityGroups.set(identity, []);
    identityGroups.get(identity).push(event);
  }
  $("gallery-description").textContent =
    `${identityGroups.size} groups · ${shown.length} events · Reference tags mark searchable images`;
  $("gallery").innerHTML =
    [...identityGroups.entries()]
      .sort(([left], [right]) => (left ?? Infinity) - (right ?? Infinity))
      .map(([identity, identityEvents]) => {
        const currentIdentity =
          identity != null && identity === e.assigned_identity;
        const windowCount = past.filter(
          (event) =>
            event.assigned_identity === identity &&
            new Date(event.event_time) > new Date(e.window_start),
        ).length;
        const references = identityEvents.filter(
          (event) => event.reference_added,
        ).length;
        const cards = identityEvents
          .map((event) => {
            const evaluation = metrics?.events[event.event_id];
            const isExtra =
              truth && annotation?.extra?.includes(event.event_id);
            const inWindow =
              new Date(event.event_time) > new Date(e.window_start);
            const color =
              truth && evaluation?.truth_group
                ? ["#1765a4", "#b86605", "#7c499c", "#338251", "#ae4257"][
                    evaluation.truth_group % 5
                  ]
                : null;
            return `<div class="event ${event === e ? "current" : ""} ${event.event_id === e.winning_reference ? "winner" : ""}"
          ${color ? `style="border-color:${color}"` : ""}>
          <img src="${image(event)}" alt="Gallery event ${safe(event.event_id)}" loading="lazy">
          <p><b>${safe(event.event_id)}</b></p>
          <p class="${event.reference_added ? "ref" : ""}">${event.reference_added ? "Reference" : "Event only"}</p>
          <p class="event-time" title="${safe(event.event_time)}">${new Date(event.event_time).toISOString().slice(11, 19)} UTC</p>
          ${inWindow ? "" : '<p class="event-time">Outside window</p>'}
          ${truth && evaluation?.truth_group ? `<p class="truthlabel">True person ${evaluation.truth_group}</p>` : ""}
          ${isExtra ? '<p class="extra">Extra in count</p>' : ""}
        </div>`;
          })
          .join("");
        return `<section class="identity-row ${currentIdentity ? "active-identity" : ""}">
        <div class="identity-label"><h3>${identity == null ? "Unresolved" : "Identity " + identity}</h3>
          ${identity == null ? `<span>${identityEvents.length} events</span>` : `<span><b>${windowCount}</b> in window</span>`}
          <span>${references} references shown</span>
          ${currentIdentity ? '<strong class="current-label">Current identity</strong>' : ""}
        </div><div class="identity-events">${cards}</div></section>`;
      })
      .join("") || '<p class="empty">No assigned events in this view.</p>';
  // Follow the changed identity inside the gallery without scrolling the page.
  const gallery = $("gallery");
  const activeRow = gallery.querySelector(".active-identity");
  if (activeRow) {
    const top = activeRow.offsetTop;
    const bottom = top + activeRow.offsetHeight;
    if (top < gallery.scrollTop) gallery.scrollTop = top;
    else if (bottom > gallery.scrollTop + gallery.clientHeight)
      gallery.scrollTop = bottom - gallery.clientHeight;
    const currentCard = activeRow.querySelector(".current");
    if (currentCard) {
      const strip = currentCard.parentElement;
      strip.scrollLeft = Math.max(
        0,
        currentCard.offsetLeft + currentCard.offsetWidth - strip.clientWidth,
      );
    }
  }
}
$("start").onclick = async () => {
  try {
    busy = true;
    controls();
    let d = await api("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: $("source").value,
        seed: Number($("seed").value),
      }),
    });
    session = d.session_id;
    localStorage.setItem("faceVelocitySession", session);
    await refresh();
  } catch (e) {
    failure(e);
  } finally {
    busy = false;
    controls();
  }
};
async function next(refreshNow = true) {
  let d = await api("/sessions/" + session + "/next", { method: "POST" });
  if (d.done) {
    running = false;
    return false;
  }
  if (refreshNow) await refresh();
  return true;
}
$("previous").onclick = () => {
  cursor = Math.max(0, cursor - 1);
  render();
};
$("next").onclick = async () => {
  if (snapshot && cursor < snapshot.events.length - 1) {
    cursor++;
    render();
    return;
  }
  try {
    busy = true;
    controls();
    await next();
  } catch (e) {
    failure(e);
  } finally {
    busy = false;
    controls();
  }
};
$("run").onclick = async () => {
  running = true;
  busy = true;
  controls();
  try {
    let n = snapshot.events.length,
      target = Math.min(Number($("target").value), Number($("target").max));
    while (running && n < target) {
      if (!(await next(false))) break;
      n++;
      if (n % 10 === 0) await refresh();
    }
    await refresh();
  } catch (e) {
    failure(e);
  } finally {
    running = false;
    busy = false;
    controls();
  }
};
$("stop").onclick = () => {
  running = false;
  controls();
};
$("source").onchange = configureSource;
$("seed").onchange = () => {
  session = null;
  clearReplay();
  controls();
};
$("cursor").oninput = () => {
  cursor = Number($("cursor").value);
  render();
};
$("truth").onchange = render;
$("gallery-filter").onchange = render;
$("upload").onsubmit = async (event) => {
  event.preventDefault();
  if (!session) {
    failure(Error("Start an upload session first."));
    return;
  }
  try {
    busy = true;
    controls();
    let form = new FormData();
    form.append("image", $("file").files[0]);
    form.append("event_id", $("event-id").value);
    form.append("event_time", $("event-time").value + "Z");
    await api("/sessions/" + session + "/events", {
      method: "POST",
      body: form,
    });
    await refresh();
    $("event-id").value = "event-" + (snapshot.events.length + 1);
  } catch (e) {
    failure(e);
  } finally {
    busy = false;
    controls();
  }
};
$("event-time").value = new Date().toISOString().slice(0, 19);

async function restore(saved) {
  if (!saved || !/^[a-f0-9]{32}$/.test(saved))
    throw Error("No saved replay in this browser.");
  busy = true;
  controls();
  let data = await api("/sessions/" + saved);
  $("source").value = data.session.source;
  $("seed").value = data.session.seed;
  await configureSource();
  session = saved;
  snapshot = data;
  cursor = Math.max(0, data.events.length - 1);
  render();
  localStorage.setItem("faceVelocitySession", session);
}
$("resume").onclick = async () => {
  try {
    await restore(localStorage.getItem("faceVelocitySession"));
  } catch (e) {
    failure(e);
  } finally {
    busy = false;
    controls();
  }
};
(async () => {
  try {
    let saved = new URLSearchParams(location.search).get("session");
    if (saved) await restore(saved);
    else await configureSource();
  } catch (e) {
    failure(e);
  } finally {
    busy = false;
    controls();
  }
})();

$("present").onclick = async () => {
  try {
    await $("presentation-shell").requestFullscreen();
  } catch {
    failure(
      Error(
        "Fullscreen is unavailable here. Open this page in Chrome, or record just the 16:9 demo panel.",
      ),
    );
  }
};
$("exit-present").onclick = () => document.exitFullscreen();
document.addEventListener("fullscreenchange", () => {
  $("exit-present").hidden = !document.fullscreenElement;
});
