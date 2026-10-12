"""HTML for the 5-hospital demo console (single template, no build step)."""

PAGE = r"""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FedShield · 5 hastane demosu</title>
  <style>
    :root { --bg:#111; --card:#1c1c1c; --fg:#eee; --muted:#9a9a9a; --line:#333; --ok:#3d9a5f; --bad:#c45c5c; --acc:#6ea8ff; }
    * { box-sizing: border-box; }
    body { margin:0; font: 15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
    header { padding:20px 24px; border-bottom:1px solid var(--line); }
    header h1 { margin:0 0 6px; font-size:20px; font-weight:600; }
    header p { margin:0; color:var(--muted); max-width:820px; }
    nav { display:flex; gap:8px; padding:12px 24px; border-bottom:1px solid var(--line); }
    nav button { background:transparent; color:var(--fg); border:1px solid var(--line); padding:6px 12px; cursor:pointer; }
    nav button.active { border-color:var(--acc); color:var(--acc); }
    main { padding:20px 24px 48px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }
    .card { background:var(--card); border:1px solid var(--line); padding:14px; }
    .card h2 { margin:0 0 4px; font-size:16px; }
    .muted { color:var(--muted); font-size:13px; }
    .ok { color:var(--ok); } .bad { color:var(--bad); }
    code { font-family: ui-monospace, Consolas, monospace; font-size:12px; }
    input[type=file] { display:block; margin:10px 0; }
    button.go { background:var(--acc); color:#111; border:0; padding:7px 12px; cursor:pointer; font-weight:600; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
    .panel { display:none; } .panel.active { display:block; }
    ol.talk { max-width:720px; }
    ol.talk li { margin:0 0 10px; }
  </style>
</head>
<body>
<header>
  <h1>FedShield · beş hastane, tek model, sıfır ham veri paylaşımı</h1>
  <p>Her kart bir hastanenin masasıdır. Yüklediğiniz CSV yalnızca o hastanenin klasörüne yazılır.
     Eğitim sırasında ağda giden şey hasta satırları değil, maskelenmiş model ağırlıklarıdır.
     Şema zorunlu: {{ input_dim }} özellik sütunu + 1 etiket (0–{{ num_classes - 1 }}).</p>
</header>
<nav>
  <button class="active" data-tab="upload">1. Veri yükle</button>
  <button data-tab="proof">2. İzolasyon kanıtı</button>
  <button data-tab="talk">3. Birine nasıl anlatılır</button>
</nav>
<main>
  <section id="upload" class="panel active">
    <p class="muted">Demo operatör ekranı: gerçek hayatta her hastane yalnızca kendi kartını görür.
       Burada beşini yan yana koyuyoruz ki kıyas net olsun. Sayılar ve SHA256 özeti var; satırlar yok.</p>
    <div class="grid">
      {% for h in hospitals %}
      <article class="card" id="card-{{ h.id }}">
        <h2>{{ h.label }}</h2>
        <p class="muted">{{ h.story }}</p>
        <p>Kimlik: <code>{{ h.id }}</code></p>
        {% if h.data.present %}
          <p>Kayıt: <strong>{{ h.data.n_samples }}</strong> · özellik {{ h.data.n_features }} · hash <code>{{ h.data.sha256 }}</code></p>
          <p class="muted">Sınıf histogramı: {{ h.data.class_hist }}</p>
          {% if h.data.out_of_range %}
          <p class="bad">Aralık dışı veya geçersiz etiket: {{ h.data.out_of_range }} kayıt histogramda yok.</p>
          {% endif %}
        {% else %}
          <p class="bad">Bu hastanede henüz partition yok.</p>
        {% endif %}
        <form data-hospital="{{ h.id }}">
          <input type="file" name="file" accept=".csv,text/csv" required/>
          <button class="go" type="submit">Bu hastaneye yükle</button>
        </form>
        <p><a href="/api/sample/{{ h.id }}" style="color:var(--acc)">Örnek CSV indir</a>
           <span class="muted">(784 özellik + etiket — rastgele Excel yüklemeyin)</span></p>
        <p class="muted status"></p>
      </article>
      {% endfor %}
    </div>
  </section>

  <section id="proof" class="panel">
    <p class="muted">Kanıt client konteynerlerinin <em>içinden</em> alınır: mount, DNS, egress log.
       Konsol diğer hastanelerin satırlarını okumaz.</p>
    <p><button class="go" id="run-proof">Kanıtı şimdi çalıştır</button>
       <a href="http://127.0.0.1:5000" style="color:var(--acc);margin-left:12px">MLflow (accuracy)</a></p>
    <p id="proof-summary" class="muted"></p>
    <table><thead><tr><th>Sonuç</th><th>Kontrol</th><th>Ayrıntı</th></tr></thead>
      <tbody id="proof-rows"></tbody></table>
    <h2 style="margin-top:24px;font-size:16px">Egress (son fit)</h2>
    <table><thead><tr><th>Hastane</th><th>payload</th><th>bytes</th><th>raw_samples_sent</th><th>SecAgg</th></tr></thead>
      <tbody id="egress-rows"></tbody></table>
  </section>

  <section id="talk" class="panel">
    <ol class="talk">
      <li><strong>Sahne:</strong> Beş hastane aynı teşhisi öğrenmek istiyor. Kayıtlar KVKK/HIPAA nedeniyle dışarı çıkamaz.</li>
      <li><strong>Veri:</strong> Her hastane kendi CSV’sini kendi kartına yükler. Hash’ler farklıdır — beş ayrı veri seti.</li>
      <li><strong>Ağ:</strong> İzolasyon kanıtında her client yalnızca <code>fl-hospital-*</code> ağına bağlıdır. Diğer hastanenin DNS adı çözülmez.</li>
      <li><strong>Disk:</strong> <code>/data/processed</code> yalnızca o hastanenin klasörüdür. Başka hastanenin <code>x.npy</code>’si o konteynerde yoktur.</li>
      <li><strong>Tel:</strong> Fit sonrası log: <code>payload_kind=model_weights</code>, <code>raw_samples_sent=0</code>. SecAgg açıkken sunucu tekil güncellemeyi de göremez.</li>
      <li><strong>Model:</strong> On round sonra MLflow’da <code>accuracy</code> yükselir — model beşinin karışımından öğrenmiştir, veri birleşmemiştir.</li>
      <li><strong>Üretim:</strong> Bu sayfa sizin laptop’unuzdadır. Gerçekte her hastane kendi VM’inde yalnızca kendi client’ını çalıştırır.</li>
    </ol>
  </section>
</main>
<script>
  document.querySelectorAll("nav button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
  document.querySelectorAll("form[data-hospital]").forEach((form) => {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const hid = form.dataset.hospital;
      const status = form.parentElement.querySelector(".status");
      const body = new FormData(form);
      status.textContent = "Yükleniyor…";
      try {
        const res = await fetch("/api/upload/" + hid, { method: "POST", body });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) { status.textContent = "Hata: " + (data.message || res.status); return; }
        status.textContent = "Yazıldı (" + data.data.n_samples + " satır, hash " + data.data.sha256 + "). Client yeniden başlatıldı: " + data.restarted;
      } catch (err) {
        status.textContent = "Konsola ulaşılamadı. Docker Desktop'ı açın, sonra .\\dev.ps1 start";
      }
    });
  });
  document.getElementById("run-proof").addEventListener("click", async () => {
    const summary = document.getElementById("proof-summary");
    summary.textContent = "Konteynerlerin içinden ölçülüyor…";
    const res = await fetch("/api/proof");
    const data = await res.json();
    summary.textContent = data.available
      ? (data.passed + " / " + data.total + " kontrol geçti")
      : (data.reason || "Docker yok");
    const rows = document.getElementById("proof-rows");
    rows.innerHTML = (data.checks || []).map((c) =>
      "<tr><td class='" + (c.ok ? "ok" : "bad") + "'>" + (c.ok ? "KANIT" : "FAIL") + "</td><td>" + c.title + "</td><td><code>" + (c.detail || "") + "</code></td></tr>"
    ).join("");
    const eg = document.getElementById("egress-rows");
    eg.innerHTML = (data.egress || []).map((e) =>
      "<tr><td>" + e.client_id + "</td><td>" + (e.payload_kind || "") + "</td><td>" + (e.nbytes || "") + "</td><td>" + e.raw_samples_sent + "</td><td>" + e.secagg_applied + "</td></tr>"
    ).join("") || "<tr><td colspan='5' class='muted'>Henüz fit yok — stack ayağa kalkınca doldurulur.</td></tr>";
  });
</script>
</body>
</html>
"""
