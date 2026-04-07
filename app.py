<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MotorMind-Predictive Maintenance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg0:#05080f;
  --bg1:#080d18;
  --bg2:#0d1525;
  --bg3:#111d33;
  --bg4:#162240;
  --cyan:#00d4ff;
  --cyan2:#0099bb;
  --cyan3:#005577;
  --green:#00ff88;
  --green2:#00c866;
  --amber:#ffaa00;
  --amber2:#cc7700;
  --red:#ff2244;
  --red2:#cc0022;
  --text:#b8d4e8;
  --text2:#6a8fa8;
  --text3:#334455;
  --border:#1a2d45;
  --border2:#0f1e30;
  --font-display:'Rajdhani',sans-serif;
  --font-mono:'JetBrains Mono',monospace;
  --glow-cyan:0 0 20px rgba(0,212,255,0.3);
  --glow-green:0 0 20px rgba(0,255,136,0.3);
  --glow-amber:0 0 20px rgba(255,170,0,0.3);
  --glow-red:0 0 20px rgba(255,34,68,0.3);
}

html,body{
  height:100%;
  background:var(--bg0);
  color:var(--text);
  font-family:var(--font-display);
  overflow-x:hidden;
}

/* ── Scanline overlay ── */
body::before{
  content:'';
  position:fixed;inset:0;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.08) 2px,
    rgba(0,0,0,0.08) 4px
  );
  pointer-events:none;
  z-index:9999;
}

/* ── Grid noise texture ── */
body::after{
  content:'';
  position:fixed;inset:0;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 40%, rgba(0,212,255,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 60%, rgba(0,255,136,0.03) 0%, transparent 50%);
  pointer-events:none;
  z-index:0;
}

/* ── Layout ── */
.shell{
  display:grid;
  grid-template-rows:56px 1fr;
  grid-template-columns:220px 1fr;
  height:100vh;
  position:relative;
  z-index:1;
}

/* ── Top bar ── */
.topbar{
  grid-column:1/-1;
  background:var(--bg1);
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  padding:0 24px;
  gap:20px;
  position:relative;
}
.topbar::after{
  content:'';
  position:absolute;
  bottom:0;left:0;right:0;
  height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:0.4;
}
.logo{
  font-size:22px;
  font-weight:700;
  color:#fff;
  letter-spacing:0.05em;
  display:flex;align-items:center;gap:10px;
}
.logo-icon{
  width:28px;height:28px;
  background:var(--cyan);
  border-radius:4px;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;
  color:var(--bg0);
  font-weight:700;
  box-shadow:var(--glow-cyan);
}
.logo span{color:var(--cyan);}
.topbar-status{
  margin-left:auto;
  display:flex;align-items:center;gap:20px;
}
.status-pill{
  display:flex;align-items:center;gap:6px;
  font-family:var(--font-mono);
  font-size:11px;
  color:var(--text2);
  letter-spacing:0.05em;
}
.status-dot{
  width:7px;height:7px;
  border-radius:50%;
  animation:pulse 2s ease-in-out infinite;
}
.status-dot.green{background:var(--green);box-shadow:0 0 8px var(--green);}
.status-dot.amber{background:var(--amber);box-shadow:0 0 8px var(--amber);}
.status-dot.red{background:var(--red);box-shadow:0 0 8px var(--red);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

.clock{
  font-family:var(--font-mono);
  font-size:13px;
  color:var(--cyan);
  letter-spacing:0.1em;
}

/* ── Sidebar ── */
.sidebar{
  background:var(--bg1);
  border-right:1px solid var(--border);
  padding:20px 0;
  overflow-y:auto;
}
.sidebar-section{
  padding:0 16px;
  margin-bottom:28px;
}
.sidebar-label{
  font-family:var(--font-mono);
  font-size:9px;
  letter-spacing:0.15em;
  color:var(--text3);
  text-transform:uppercase;
  padding:0 8px;
  margin-bottom:8px;
}
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:9px 12px;
  border-radius:6px;
  cursor:pointer;
  font-size:14px;
  font-weight:500;
  color:var(--text2);
  letter-spacing:0.03em;
  transition:all 0.15s;
  margin-bottom:2px;
  border:1px solid transparent;
}
.nav-item:hover{
  background:var(--bg3);
  color:var(--text);
  border-color:var(--border);
}
.nav-item.active{
  background:rgba(0,212,255,0.08);
  color:var(--cyan);
  border-color:rgba(0,212,255,0.2);
}
.nav-item.active .nav-icon{color:var(--cyan);}
.nav-icon{font-size:15px;width:18px;text-align:center;}

.motor-cards{display:flex;flex-direction:column;gap:8px;}
.motor-card{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:8px;
  padding:10px 12px;
  cursor:pointer;
  transition:all 0.15s;
}
.motor-card:hover{border-color:var(--cyan3);}
.motor-card.selected{border-color:var(--cyan);background:rgba(0,212,255,0.06);}
.motor-card-id{
  font-family:var(--font-mono);
  font-size:12px;
  font-weight:500;
  color:var(--text);
  margin-bottom:4px;
}
.motor-card-status{
  font-size:11px;
  color:var(--text2);
  display:flex;align-items:center;gap:5px;
}
.motor-health-bar{
  height:3px;
  background:var(--bg4);
  border-radius:2px;
  margin-top:7px;
  overflow:hidden;
}
.motor-health-fill{
  height:100%;
  border-radius:2px;
  transition:width 1s ease;
}

/* ── Main content ── */
.main{
  background:var(--bg0);
  overflow-y:auto;
  padding:24px;
  display:flex;
  flex-direction:column;
  gap:20px;
}

/* ── Section header ── */
.section-header{
  display:flex;align-items:center;gap:12px;
  margin-bottom:4px;
}
.section-title{
  font-size:18px;
  font-weight:600;
  color:#fff;
  letter-spacing:0.03em;
}
.section-sub{
  font-size:12px;
  color:var(--text2);
  font-family:var(--font-mono);
  margin-left:auto;
}
.tag{
  font-family:var(--font-mono);
  font-size:9px;
  font-weight:500;
  letter-spacing:0.12em;
  text-transform:uppercase;
  padding:3px 8px;
  border-radius:3px;
}
.tag-cyan{background:rgba(0,212,255,0.12);color:var(--cyan);border:1px solid rgba(0,212,255,0.2);}
.tag-green{background:rgba(0,255,136,0.12);color:var(--green);border:1px solid rgba(0,255,136,0.2);}
.tag-amber{background:rgba(255,170,0,0.12);color:var(--amber);border:1px solid rgba(255,170,0,0.2);}
.tag-red{background:rgba(255,34,68,0.12);color:var(--red);border:1px solid rgba(255,34,68,0.2);}

/* ── KPI row ── */
.kpi-row{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:12px;
}
.kpi-card{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:10px;
  padding:16px 18px;
  position:relative;
  overflow:hidden;
  transition:border-color 0.2s;
}
.kpi-card::before{
  content:'';
  position:absolute;
  top:0;left:0;right:0;
  height:2px;
}
.kpi-card.cyan::before{background:linear-gradient(90deg,transparent,var(--cyan),transparent);}
.kpi-card.green::before{background:linear-gradient(90deg,transparent,var(--green),transparent);}
.kpi-card.amber::before{background:linear-gradient(90deg,transparent,var(--amber),transparent);}
.kpi-card.red::before{background:linear-gradient(90deg,transparent,var(--red),transparent);}
.kpi-card:hover{border-color:var(--border2);}
.kpi-label{
  font-family:var(--font-mono);
  font-size:9px;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:var(--text2);
  margin-bottom:10px;
}
.kpi-value{
  font-size:28px;
  font-weight:700;
  line-height:1;
  margin-bottom:4px;
}
.kpi-card.cyan .kpi-value{color:var(--cyan);}
.kpi-card.green .kpi-value{color:var(--green);}
.kpi-card.amber .kpi-value{color:var(--amber);}
.kpi-card.red .kpi-value{color:var(--red);}
.kpi-delta{
  font-family:var(--font-mono);
  font-size:10px;
  color:var(--text2);
}
.kpi-delta.up{color:var(--red);}
.kpi-delta.down{color:var(--green);}

/* ── Grid 2-col ── */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}

/* ── Panel ── */
.panel{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:10px;
  overflow:hidden;
}
.panel-header{
  padding:14px 18px;
  border-bottom:1px solid var(--border2);
  display:flex;align-items:center;gap:10px;
}
.panel-title{
  font-size:13px;
  font-weight:600;
  color:var(--text);
  letter-spacing:0.05em;
  text-transform:uppercase;
}
.panel-body{padding:18px;}

/* ── Gauge ── */
.gauge-wrap{
  display:flex;
  flex-direction:column;
  align-items:center;
  padding:20px 10px 10px;
}
.gauge-svg{width:180px;height:100px;overflow:visible;}
.gauge-value{
  font-family:var(--font-mono);
  font-size:32px;
  font-weight:500;
  text-anchor:middle;
  dominant-baseline:central;
}
.gauge-unit{
  font-family:var(--font-mono);
  font-size:11px;
  text-anchor:middle;
  fill:var(--text2);
}
.gauge-label{
  font-family:var(--font-mono);
  font-size:10px;
  letter-spacing:0.1em;
  text-transform:uppercase;
  color:var(--text2);
  margin-top:8px;
}

/* ── Sensor row ── */
.sensor-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:10px;
}
.sensor-item{
  background:var(--bg3);
  border:1px solid var(--border2);
  border-radius:8px;
  padding:12px 14px;
  transition:border-color 0.2s;
}
.sensor-item:hover{border-color:var(--border);}
.sensor-item.warn{border-color:rgba(255,170,0,0.4);}
.sensor-item.crit{border-color:rgba(255,34,68,0.4);}
.sensor-name{
  font-family:var(--font-mono);
  font-size:9px;
  letter-spacing:0.1em;
  text-transform:uppercase;
  color:var(--text2);
  margin-bottom:8px;
}
.sensor-val{
  font-family:var(--font-mono);
  font-size:20px;
  font-weight:500;
  color:#fff;
  line-height:1;
}
.sensor-unit{font-size:11px;color:var(--text2);margin-left:3px;}
.sensor-bar{
  height:3px;
  background:var(--bg4);
  border-radius:2px;
  margin-top:8px;
  overflow:hidden;
}
.sensor-bar-fill{
  height:100%;border-radius:2px;
  transition:width 1.2s ease;
}

/* ── Trend chart ── */
.chart-canvas-wrap{
  width:100%;
  height:160px;
  position:relative;
}
canvas.trend{width:100%!important;height:160px!important;}

/* ── Failure probability chart ── */
.prob-chart-wrap{
  width:100%;height:200px;
  position:relative;
}
canvas.prob{width:100%!important;height:200px!important;}

/* ── Failure mode badge ── */
.failure-badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:10px 16px;
  border-radius:8px;
  font-size:15px;
  font-weight:600;
  letter-spacing:0.04em;
  border:1px solid;
}
.failure-badge.normal{
  background:rgba(0,255,136,0.08);
  color:var(--green);
  border-color:rgba(0,255,136,0.25);
}
.failure-badge.warning{
  background:rgba(255,170,0,0.08);
  color:var(--amber);
  border-color:rgba(255,170,0,0.25);
}
.failure-badge.critical{
  background:rgba(255,34,68,0.08);
  color:var(--red);
  border-color:rgba(255,34,68,0.25);
  animation:blink-border 1.5s ease-in-out infinite;
}
@keyframes blink-border{
  0%,100%{box-shadow:none;}
  50%{box-shadow:var(--glow-red);}
}

/* ── Alert strip ── */
.alert-strip{
  background:rgba(255,34,68,0.06);
  border:1px solid rgba(255,34,68,0.25);
  border-radius:8px;
  padding:12px 16px;
  display:flex;align-items:center;gap:12px;
  font-size:13px;
  color:var(--red);
  font-weight:500;
}
.alert-icon{font-size:16px;flex-shrink:0;}

/* ── Input table ── */
.input-table{width:100%;border-collapse:collapse;}
.input-table th{
  font-family:var(--font-mono);
  font-size:9px;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:var(--text2);
  padding:8px 10px;
  text-align:left;
  border-bottom:1px solid var(--border2);
}
.input-table td{
  padding:6px 10px;
  border-bottom:1px solid var(--border2);
  font-family:var(--font-mono);
  font-size:12px;
  color:var(--text);
}
.input-table tr:last-child td{border-bottom:none;}
.input-table tr:hover td{background:rgba(255,255,255,0.02);}
input[type=number]{
  background:var(--bg3);
  border:1px solid var(--border);
  border-radius:4px;
  color:var(--cyan);
  font-family:var(--font-mono);
  font-size:12px;
  padding:5px 8px;
  width:100px;
  outline:none;
  transition:border-color 0.15s;
}
input[type=number]:focus{border-color:var(--cyan);}
input[type=number]::-webkit-inner-spin-button{opacity:0.3;}

/* ── Button ── */
.btn{
  display:inline-flex;align-items:center;gap:8px;
  padding:10px 20px;
  border-radius:6px;
  font-family:var(--font-display);
  font-size:14px;
  font-weight:600;
  letter-spacing:0.06em;
  cursor:pointer;
  transition:all 0.15s;
  border:none;
  text-transform:uppercase;
}
.btn-primary{
  background:var(--cyan);
  color:var(--bg0);
  box-shadow:0 0 20px rgba(0,212,255,0.2);
}
.btn-primary:hover{
  background:#33ddff;
  box-shadow:0 0 30px rgba(0,212,255,0.4);
  transform:translateY(-1px);
}
.btn-primary:active{transform:translateY(0);}
.btn-ghost{
  background:transparent;
  color:var(--text2);
  border:1px solid var(--border);
}
.btn-ghost:hover{
  background:var(--bg3);
  color:var(--text);
  border-color:var(--cyan3);
}

/* ── SHAP bar ── */
.shap-row{
  display:flex;align-items:center;gap:10px;
  margin-bottom:8px;
}
.shap-label{
  font-family:var(--font-mono);
  font-size:10px;
  color:var(--text2);
  width:180px;
  flex-shrink:0;
  text-align:right;
}
.shap-bar-bg{
  flex:1;
  height:8px;
  background:var(--bg4);
  border-radius:4px;
  overflow:hidden;
}
.shap-bar-fill{
  height:100%;
  border-radius:4px;
  transition:width 1s ease;
}
.shap-bar-fill.pos{background:var(--red);}
.shap-bar-fill.neg{background:var(--green);}
.shap-val{
  font-family:var(--font-mono);
  font-size:10px;
  width:50px;
  text-align:right;
}

/* ── Class prob bars ── */
.class-row{
  display:flex;align-items:center;gap:10px;
  margin-bottom:10px;
}
.class-name{
  font-family:var(--font-mono);
  font-size:10px;
  color:var(--text2);
  width:140px;
  flex-shrink:0;
}
.class-bar-bg{
  flex:1;height:10px;
  background:var(--bg4);
  border-radius:5px;
  overflow:hidden;
}
.class-bar-fill{
  height:100%;border-radius:5px;
  transition:width 1s ease;
}
.class-pct{
  font-family:var(--font-mono);
  font-size:10px;
  width:40px;text-align:right;
  color:var(--text);
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg1);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:var(--cyan3);}

/* ── Page transitions ── */
.page{display:none;animation:fadeIn 0.25s ease;}
.page.active{display:flex;flex-direction:column;gap:20px;}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

/* ── Divider ── */
.divider{
  height:1px;
  background:linear-gradient(90deg,transparent,var(--border),transparent);
}

/* ── Health ring ── */
.health-ring-wrap{
  display:flex;flex-direction:column;align-items:center;
  padding:10px;
}

/* ── Tooltip ── */
.tooltip-wrap{position:relative;display:inline-block;}
.tooltip-wrap .tip{
  display:none;
  position:absolute;
  bottom:calc(100% + 6px);
  left:50%;transform:translateX(-50%);
  background:var(--bg4);
  border:1px solid var(--border);
  border-radius:6px;
  padding:6px 10px;
  font-family:var(--font-mono);
  font-size:10px;
  color:var(--text);
  white-space:nowrap;
  z-index:100;
}
.tooltip-wrap:hover .tip{display:block;}

/* ── Responsive ── */
@media(max-width:900px){
  .shell{grid-template-columns:1fr;}
  .sidebar{display:none;}
  .kpi-row{grid-template-columns:1fr 1fr;}
  .grid-2,.grid-3{grid-template-columns:1fr;}
  .sensor-grid{grid-template-columns:1fr 1fr;}
}
</style>
</head>
<body>

<div class="shell">

  <!-- ── Top bar ── -->
  <header class="topbar">
    <div class="logo">
      <div class="logo-icon">M</div>
      Motor<span>Mind</span>
    </div>
    <div style="font-family:var(--font-mono);font-size:10px;color:var(--text3);letter-spacing:0.1em;margin-left:16px;">
      PREDICTIVE MAINTENANCE SYSTEM v2.0
    </div>
    <div class="topbar-status">
      <div class="status-pill">
        <div class="status-dot green" id="sys-dot"></div>
        <span id="sys-status">SYSTEM ONLINE</span>
      </div>
      <div class="status-pill">
        <div class="status-dot" id="model-dot" style="background:var(--cyan);box-shadow:0 0 8px var(--cyan);"></div>
        MODEL ACTIVE
      </div>
      <div class="clock" id="clock">--:--:--</div>
    </div>
  </header>

  <!-- ── Sidebar ── -->
  <nav class="sidebar">
    <div class="sidebar-section">
      <div class="sidebar-label">Navigation</div>
      <div class="nav-item active" onclick="showPage('dashboard',this)">
        <span class="nav-icon">◈</span> Dashboard
      </div>
      <div class="nav-item" onclick="showPage('predict',this)">
        <span class="nav-icon">◎</span> Predict
      </div>
      <div class="nav-item" onclick="showPage('trends',this)">
        <span class="nav-icon">◻</span> Trend Analysis
      </div>
      <div class="nav-item" onclick="showPage('model',this)">
        <span class="nav-icon">◆</span> Model Info
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Fleet — 3 motors</div>
      <div class="motor-cards">
        <div class="motor-card selected" onclick="selectMotor('MOTOR-01',this)">
          <div class="motor-card-id">MOTOR-01</div>
          <div class="motor-card-status">
            <div class="status-dot green" style="width:6px;height:6px;"></div>
            Normal · 1448 RPM
          </div>
          <div class="motor-health-bar">
            <div class="motor-health-fill" style="width:88%;background:var(--green);"></div>
          </div>
        </div>
        <div class="motor-card" onclick="selectMotor('MOTOR-02',this)">
          <div class="motor-card-id">MOTOR-02</div>
          <div class="motor-card-status">
            <div class="status-dot amber" style="width:6px;height:6px;"></div>
            Degrading · 1382 RPM
          </div>
          <div class="motor-health-bar">
            <div class="motor-health-fill" style="width:52%;background:var(--amber);"></div>
          </div>
        </div>
        <div class="motor-card" onclick="selectMotor('MOTOR-03',this)">
          <div class="motor-card-id">MOTOR-03</div>
          <div class="motor-card-status">
            <div class="status-dot red" style="width:6px;height:6px;animation:pulse 0.8s ease-in-out infinite;"></div>
            ALERT · 1210 RPM
          </div>
          <div class="motor-health-bar">
            <div class="motor-health-fill" style="width:21%;background:var(--red);"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">System</div>
      <div style="font-family:var(--font-mono);font-size:10px;color:var(--text2);padding:0 8px;line-height:2;">
        <div>Model: <span style="color:var(--cyan);">LightGBM</span></div>
        <div>Features: <span style="color:var(--cyan);">91</span></div>
        <div>AUC: <span style="color:var(--green);">0.9987</span></div>
        <div>Windows: <span style="color:var(--cyan);">3 · 6 · 12</span></div>
        <div>Sampling: <span style="color:var(--cyan);">10 min</span></div>
      </div>
    </div>
  </nav>

  <!-- ── Main ── -->
  <main class="main">

    <!-- ════════════════════ DASHBOARD PAGE ════════════════════ -->
    <div class="page active" id="page-dashboard">

      <div class="section-header">
        <div class="section-title">Fleet Overview</div>
        <div id="selected-motor-badge" class="tag tag-cyan">MOTOR-01</div>
        <div class="section-sub" id="last-updated">Last updated: just now</div>
      </div>

      <!-- KPI row -->
      <div class="kpi-row">
        <div class="kpi-card cyan">
          <div class="kpi-label">Failure Probability</div>
          <div class="kpi-value" id="kpi-prob">8.2<span style="font-size:16px;">%</span></div>
          <div class="kpi-delta down" id="kpi-prob-delta">▼ 1.4% vs last hour</div>
        </div>
        <div class="kpi-card green">
          <div class="kpi-label">Health Score</div>
          <div class="kpi-value" id="kpi-health">91.8<span style="font-size:16px;">/100</span></div>
          <div class="kpi-delta" id="kpi-health-delta" style="color:var(--text2);">Rolling 12-reading avg</div>
        </div>
        <div class="kpi-card amber">
          <div class="kpi-label">Predicted Mode</div>
          <div class="kpi-value" style="font-size:16px;padding-top:4px;" id="kpi-mode">Normal</div>
          <div class="kpi-delta" style="color:var(--text2);" id="kpi-mode-conf">Confidence 94.1%</div>
        </div>
        <div class="kpi-card cyan">
          <div class="kpi-label">Active Alerts</div>
          <div class="kpi-value" id="kpi-alerts">0</div>
          <div class="kpi-delta" style="color:var(--text2);" id="kpi-alerts-sub">All parameters nominal</div>
        </div>
      </div>

      <!-- Gauge + Sensors -->
      <div class="grid-2">

        <!-- Failure probability gauge -->
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Failure Probability Gauge</div>
            <div class="tag tag-cyan" style="margin-left:auto;" id="gauge-status-tag">NORMAL</div>
          </div>
          <div class="panel-body" style="display:flex;align-items:center;justify-content:center;gap:30px;flex-wrap:wrap;">
            <div class="gauge-wrap">
              <svg class="gauge-svg" viewBox="0 0 180 100">
                <defs>
                  <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#00ff88"/>
                    <stop offset="50%" stop-color="#ffaa00"/>
                    <stop offset="100%" stop-color="#ff2244"/>
                  </linearGradient>
                </defs>
                <!-- Track -->
                <path d="M 15 95 A 75 75 0 0 1 165 95" fill="none" stroke="#162240" stroke-width="14" stroke-linecap="round"/>
                <!-- Gradient track fill -->
                <path d="M 15 95 A 75 75 0 0 1 165 95" fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round" opacity="0.15"/>
                <!-- Active arc -->
                <path id="gauge-arc" d="M 15 95 A 75 75 0 0 1 165 95" fill="none" stroke="#00ff88" stroke-width="14" stroke-linecap="round"
                  stroke-dasharray="235.6" stroke-dashoffset="216.7"/>
                <!-- Needle -->
                <line id="gauge-needle" x1="90" y1="95" x2="25" y2="88"
                  stroke="#fff" stroke-width="2" stroke-linecap="round" opacity="0.9"/>
                <circle cx="90" cy="95" r="5" fill="#fff" opacity="0.9"/>
                <!-- Value -->
                <text class="gauge-value" id="gauge-text" x="90" y="72" fill="#00ff88">8.2%</text>
                <text class="gauge-unit" x="90" y="86">FAILURE PROB.</text>
              </svg>
              <div class="gauge-label" id="gauge-label-text">NORMAL OPERATION</div>
            </div>

            <!-- Health ring -->
            <div class="health-ring-wrap">
              <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="50" fill="none" stroke="#162240" stroke-width="10"/>
                <circle id="health-ring" cx="60" cy="60" r="50" fill="none" stroke="#00ff88" stroke-width="10"
                  stroke-dasharray="314.16" stroke-dashoffset="37.7"
                  stroke-linecap="round" transform="rotate(-90 60 60)"
                  style="transition:stroke-dashoffset 1s ease,stroke 0.5s ease;"/>
                <text x="60" y="56" text-anchor="middle" dominant-baseline="central"
                  font-family="'JetBrains Mono',monospace" font-size="22" font-weight="500" fill="#00ff88" id="health-ring-val">91.8</text>
                <text x="60" y="74" text-anchor="middle"
                  font-family="'JetBrains Mono',monospace" font-size="9" fill="#6a8fa8" letter-spacing="1">HEALTH</text>
              </svg>
            </div>
          </div>
        </div>

        <!-- Failure class probabilities -->
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Class Probabilities</div>
            <div class="tag tag-cyan" style="margin-left:auto;">MULTI-CLASS</div>
          </div>
          <div class="panel-body" id="class-probs-panel">
            <!-- Rendered by JS -->
          </div>
        </div>
      </div>

      <!-- Sensor grid -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Live Sensor Readings</div>
          <div class="tag tag-green" style="margin-left:auto;">7 CHANNELS</div>
        </div>
        <div class="panel-body">
          <div class="sensor-grid" id="sensor-grid">
            <!-- Rendered by JS -->
          </div>
        </div>
      </div>

      <!-- Probability trend chart -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Failure Probability — Time Series</div>
          <div style="margin-left:auto;display:flex;gap:8px;">
            <span class="tag tag-red">HIGH RISK ≥60%</span>
            <span class="tag tag-amber">WATCH ≥25%</span>
          </div>
        </div>
        <div class="panel-body">
          <div class="prob-chart-wrap">
            <canvas class="prob" id="prob-chart"></canvas>
          </div>
        </div>
      </div>

    </div><!-- /page-dashboard -->


    <!-- ════════════════════ PREDICT PAGE ════════════════════ -->
    <div class="page" id="page-predict">

      <div class="section-header">
        <div class="section-title">Manual Prediction</div>
        <div class="tag tag-cyan">ROLLING WINDOW INPUT</div>
      </div>

      <div class="panel" style="padding:18px;">
        <div style="font-family:var(--font-mono);font-size:11px;color:var(--text2);margin-bottom:16px;line-height:1.8;background:var(--bg3);padding:12px 14px;border-radius:6px;border-left:3px solid var(--cyan);">
          Enter sensor readings from oldest → newest. Each row = one 10-minute reading.<br>
          The model uses the <span style="color:var(--cyan);">trend across all rows</span>, not just the latest value. More rows = better prediction.
        </div>

        <div style="display:flex;gap:12px;margin-bottom:16px;align-items:center;flex-wrap:wrap;">
          <div>
            <label style="font-family:var(--font-mono);font-size:10px;color:var(--text2);display:block;margin-bottom:4px;letter-spacing:0.08em;">MOTOR ID</label>
            <select id="pred-motor-id" style="background:var(--bg3);border:1px solid var(--border);color:var(--text);font-family:var(--font-mono);font-size:12px;padding:7px 10px;border-radius:5px;outline:none;">
              <option>MOTOR-01</option><option>MOTOR-02</option><option>MOTOR-03</option>
            </select>
          </div>
          <div>
            <label style="font-family:var(--font-mono);font-size:10px;color:var(--text2);display:block;margin-bottom:4px;letter-spacing:0.08em;">READINGS (rows)</label>
            <input type="number" id="n-readings" value="6" min="1" max="12" style="width:80px;" onchange="buildInputTable()">
          </div>
          <button class="btn btn-ghost" onclick="fillDefaults()" style="margin-top:16px;">
            Fill defaults
          </button>
          <button class="btn btn-primary" onclick="runPrediction()" style="margin-top:16px;">
            ◎ Run Prediction
          </button>
        </div>

        <div style="overflow-x:auto;">
          <table class="input-table" id="input-table">
            <!-- Built by JS -->
          </table>
        </div>
      </div>

      <!-- Results -->
      <div id="pred-results" style="display:none;">
        <div class="divider"></div>
        <div class="section-header" style="margin-top:4px;">
          <div class="section-title">Prediction Result</div>
          <div class="tag tag-cyan" id="res-motor-label">MOTOR-01</div>
        </div>

        <div class="kpi-row">
          <div class="kpi-card" id="res-prob-card" style="border-top-color:var(--green);">
            <div class="kpi-label">Failure Probability</div>
            <div class="kpi-value" id="res-prob-val" style="color:var(--green);">—</div>
          </div>
          <div class="kpi-card cyan">
            <div class="kpi-label">Predicted Mode</div>
            <div class="kpi-value" id="res-mode-val" style="font-size:15px;padding-top:5px;">—</div>
          </div>
          <div class="kpi-card green">
            <div class="kpi-label">Health Score</div>
            <div class="kpi-value" id="res-health-val" style="color:var(--green);">—</div>
          </div>
          <div class="kpi-card amber">
            <div class="kpi-label">Confidence</div>
            <div class="kpi-value" id="res-conf-val" style="color:var(--amber);">—</div>
          </div>
        </div>

        <div class="grid-2">
          <div class="panel">
            <div class="panel-header"><div class="panel-title">Maintenance Action</div></div>
            <div class="panel-body" id="res-action-panel"></div>
          </div>
          <div class="panel">
            <div class="panel-header"><div class="panel-title">Class Probabilities</div></div>
            <div class="panel-body" id="res-class-panel"></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header"><div class="panel-title">Trend Insight</div></div>
          <div class="panel-body" id="res-trend-panel">
            <div class="grid-2" id="trend-charts"></div>
          </div>
        </div>
      </div>

    </div><!-- /page-predict -->


    <!-- ════════════════════ TRENDS PAGE ════════════════════ -->
    <div class="page" id="page-trends">
      <div class="section-header">
        <div class="section-title">Trend Analysis</div>
        <div class="tag tag-cyan">TIME-SERIES VIEW</div>
      </div>
      <div class="panel">
        <div class="panel-header"><div class="panel-title">Failure Probability — All Motors</div></div>
        <div class="panel-body">
          <div class="prob-chart-wrap" style="height:260px;">
            <canvas id="all-motors-chart" style="width:100%!important;height:260px!important;"></canvas>
          </div>
        </div>
      </div>
      <div class="grid-3" id="sensor-trend-panels">
        <!-- Built by JS -->
      </div>
    </div>


    <!-- ════════════════════ MODEL PAGE ════════════════════ -->
    <div class="page" id="page-model">
      <div class="section-header">
        <div class="section-title">Model Information</div>
        <div class="tag tag-cyan">LightGBM · MULTICLASS</div>
      </div>

      <div class="kpi-row">
        <div class="kpi-card cyan"><div class="kpi-label">ROC-AUC</div><div class="kpi-value">0.9987</div></div>
        <div class="kpi-card green"><div class="kpi-label">Accuracy</div><div class="kpi-value">96<span style="font-size:16px;">%</span></div></div>
        <div class="kpi-card amber"><div class="kpi-label">Features</div><div class="kpi-value">91</div></div>
        <div class="kpi-card cyan"><div class="kpi-label">Classes</div><div class="kpi-value">6</div></div>
      </div>

      <div class="grid-2">
        <div class="panel">
          <div class="panel-header"><div class="panel-title">Feature Engineering</div></div>
          <div class="panel-body" style="font-family:var(--font-mono);font-size:11px;line-height:2.2;color:var(--text2);">
            <div>Raw sensors: <span style="color:var(--cyan);">7 channels</span></div>
            <div>Window sizes: <span style="color:var(--cyan);">3 · 6 · 12 readings</span></div>
            <div>= 30 min · 1 hr · 2 hr</div>
            <div style="margin-top:8px;color:var(--text);">Per sensor × per window:</div>
            <div>Rolling mean <span style="color:var(--green);">→ trend level</span></div>
            <div>Rolling std <span style="color:var(--amber);">→ instability</span></div>
            <div>Rolling max <span style="color:var(--red);">→ worst case</span></div>
            <div>Delta (current − mean) <span style="color:var(--cyan);">→ rate of change</span></div>
            <div style="margin-top:8px;color:var(--text);">Total: 7 × 4 × 3 + 7 = <span style="color:var(--cyan);">91 features</span></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header"><div class="panel-title">Failure Classes</div></div>
          <div class="panel-body">
            <div id="class-detail-list">
              <!-- Built by JS -->
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header"><div class="panel-title">Per-class Performance</div></div>
        <div class="panel-body">
          <div id="perf-table"></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header"><div class="panel-title">Limitations — Important for Report</div></div>
        <div class="panel-body" style="font-family:var(--font-mono);font-size:11px;line-height:2;color:var(--text2);">
          <div style="color:var(--amber);">⚠ Training data is physics-based synthetic — not real factory measurements</div>
          <div style="color:var(--amber);">⚠ Failure thresholds reflect simulated distributions, not empirical factory data</div>
          <div style="color:var(--text2);">◈ Rolling window methodology is valid and transferable to real data</div>
          <div style="color:var(--text2);">◈ Model architecture is production-ready — requires retraining on real sensor logs</div>
          <div style="color:var(--green);">✓ CWRU bearing dataset recommended as next real-data training source</div>
          <div style="color:var(--green);">✓ Multi-class labels (HDF/OSF/TWF/PWF/VBF) are clinically meaningful</div>
        </div>
      </div>
    </div><!-- /page-model -->

  </main>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
// ── Data & State ───────────────────────────────────────────
const MOTORS = {
  'MOTOR-01': {
    prob: 0.082, health: 91.8, mode: 'Normal', rpm: 1448,
    sensors: {
      Winding_Temp_K: {val:328.4, max:430, warn:380, unit:'K'},
      Bearing_Temp_K: {val:322.1, max:410, warn:360, unit:'K'},
      Rotational_Speed_RPM:{val:1448, max:1800, warn:1600, unit:'RPM'},
      Torque_Nm: {val:36.2, max:120, warn:70, unit:'Nm'},
      Vibration_mmps:{val:1.8, max:18, warn:7.1, unit:'mm/s'},
      Current_Imbalance_pct:{val:1.4, max:15, warn:5, unit:'%'},
      Insulation_Resistance_MOhm:{val:212, max:520, warn:50, unit:'MΩ'},
    },
    probs:{Normal:0.918,'HDF - Heat':0.04,'OSF - Overspeed':0.01,'TWF - Wear':0.012,'PWF - Electrical':0.01,'VBF - Vibration':0.01},
    history:[4.2,5.1,6.8,7.2,8.0,8.2,7.9,8.2]
  },
  'MOTOR-02': {
    prob: 0.48, health: 52.0, mode: 'HDF - Heat', rpm: 1382,
    sensors: {
      Winding_Temp_K:{val:385.2, max:430, warn:380, unit:'K'},
      Bearing_Temp_K:{val:362.8, max:410, warn:360, unit:'K'},
      Rotational_Speed_RPM:{val:1382, max:1800, warn:1600, unit:'RPM'},
      Torque_Nm:{val:54.1, max:120, warn:70, unit:'Nm'},
      Vibration_mmps:{val:3.9, max:18, warn:7.1, unit:'mm/s'},
      Current_Imbalance_pct:{val:2.8, max:15, warn:5, unit:'%'},
      Insulation_Resistance_MOhm:{val:88, max:520, warn:50, unit:'MΩ'},
    },
    probs:{Normal:0.52,'HDF - Heat':0.38,'OSF - Overspeed':0.04,'TWF - Wear':0.02,'PWF - Electrical':0.02,'VBF - Vibration':0.02},
    history:[18,22,28,35,41,43,46,48]
  },
  'MOTOR-03': {
    prob: 0.79, health: 21.0, mode: 'VBF - Vibration', rpm: 1210,
    sensors: {
      Winding_Temp_K:{val:349.0, max:430, warn:380, unit:'K'},
      Bearing_Temp_K:{val:344.2, max:410, warn:360, unit:'K'},
      Rotational_Speed_RPM:{val:1210, max:1800, warn:1600, unit:'RPM'},
      Torque_Nm:{val:62.8, max:120, warn:70, unit:'Nm'},
      Vibration_mmps:{val:9.4, max:18, warn:7.1, unit:'mm/s'},
      Current_Imbalance_pct:{val:6.2, max:15, warn:5, unit:'%'},
      Insulation_Resistance_MOhm:{val:42, max:520, warn:50, unit:'MΩ'},
    },
    probs:{Normal:0.21,'HDF - Heat':0.08,'OSF - Overspeed':0.06,'TWF - Wear':0.05,'PWF - Electrical':0.07,'VBF - Vibration':0.53},
    history:[38,44,51,58,65,70,76,79]
  }
};

const ACTIONS = {
  'Normal':'All parameters within normal limits. Continue standard monitoring intervals.',
  'HDF - Heat':'Heat fault detected. Check cooling system, clean air filters, inspect cooling fan, verify ambient temperature.',
  'OSF - Overspeed':'Overspeed / load fault. Inspect supply voltage, check load coupling, verify VFD settings.',
  'TWF - Wear':'Wear fault. Schedule bearing replacement, check lubrication schedule, inspect shaft seal.',
  'PWF - Electrical':'Electrical fault. Inspect phase supply balance, check winding insulation resistance, test contactors.',
  'VBF - Vibration':'Vibration fault. Check shaft alignment, rotor balance, mounting bolts, and bearing condition.',
};

const CLASS_COLORS = {
  'Normal':'#00ff88',
  'HDF - Heat':'#ff2244',
  'OSF - Overspeed':'#ffaa00',
  'TWF - Wear':'#ff7700',
  'PWF - Electrical':'#aa44ff',
  'VBF - Vibration':'#00d4ff',
};

let currentMotor = 'MOTOR-01';
let probChart = null;
let allMotorsChart = null;
let trendCharts = {};

// ── Clock ──────────────────────────────────────────────────
function updateClock(){
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toTimeString().slice(0,8);
}
setInterval(updateClock, 1000);
updateClock();

// ── Page nav ───────────────────────────────────────────────
function showPage(id, el){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  el.classList.add('active');
  if(id==='trends') buildTrendPage();
  if(id==='model')  buildModelPage();
}

// ── Motor selection ────────────────────────────────────────
function selectMotor(id, el){
  currentMotor = id;
  document.querySelectorAll('.motor-card').forEach(c=>c.classList.remove('selected'));
  el.classList.add('selected');
  renderDashboard();
}

// ── Gauge helpers ──────────────────────────────────────────
function probToColor(p){
  if(p>=0.6) return '#ff2244';
  if(p>=0.25) return '#ffaa00';
  return '#00ff88';
}

function updateGauge(p){
  const pct = Math.min(p, 1);
  const arcLen = 235.6;
  const offset = arcLen - arcLen * pct;
  const col = probToColor(p);
  const arc = document.getElementById('gauge-arc');
  const needle = document.getElementById('gauge-needle');
  const txt = document.getElementById('gauge-text');
  const lbl = document.getElementById('gauge-label-text');
  const tag = document.getElementById('gauge-status-tag');

  arc.setAttribute('stroke-dashoffset', offset);
  arc.setAttribute('stroke', col);

  // Needle rotation: -90deg = 0%, 90deg = 100%
  const angle = -90 + pct * 180;
  const rad = angle * Math.PI / 180;
  const nx = 90 + 65 * Math.cos(rad);
  const ny = 95 + 65 * Math.sin(rad);
  needle.setAttribute('x2', nx);
  needle.setAttribute('y2', ny);

  txt.textContent = (p*100).toFixed(1)+'%';
  txt.setAttribute('fill', col);

  const labels = ['NORMAL OPERATION','DEGRADING — MONITOR','FAILURE PREDICTED'];
  const tagClasses = ['tag-green','tag-amber','tag-red'];
  const tagTexts = ['NORMAL','DEGRADING','CRITICAL'];
  const lvl = p>=0.6?2:p>=0.25?1:0;
  lbl.textContent = labels[lvl];
  tag.className = 'tag '+tagClasses[lvl];
  tag.textContent = tagTexts[lvl];
}

function updateHealthRing(h){
  const pct = h/100;
  const circ = 314.16;
  const offset = circ - circ*pct;
  const col = h>60?'#00ff88':h>30?'#ffaa00':'#ff2244';
  const ring = document.getElementById('health-ring');
  ring.setAttribute('stroke-dashoffset', offset);
  ring.setAttribute('stroke', col);
  document.getElementById('health-ring-val').textContent = h.toFixed(1);
  document.getElementById('health-ring-val').setAttribute('fill', col);
}

// ── Sensor grid ────────────────────────────────────────────
function renderSensorGrid(motorId){
  const m = MOTORS[motorId];
  const grid = document.getElementById('sensor-grid');
  grid.innerHTML = '';
  Object.entries(m.sensors).forEach(([key, s])=>{
    const pct = Math.min(s.val/s.max*100, 100);
    const isCrit = s.val > s.warn;
    const isWarn = s.val > s.warn * 0.85;
    const barColor = isCrit?'var(--red)':isWarn?'var(--amber)':'var(--cyan)';
    const cls = isCrit?'crit':isWarn?'warn':'';
    const label = key.replace(/_/g,' ').replace('Rotational Speed RPM','RPM')
                     .replace('Current Imbalance pct','Current Imbalance')
                     .replace('Insulation Resistance MOhm','Insulation R')
                     .replace('Winding Temp K','Winding Temp')
                     .replace('Bearing Temp K','Bearing Temp')
                     .replace('Vibration mmps','Vibration')
                     .replace('Torque Nm','Torque');
    grid.innerHTML += `
      <div class="sensor-item ${cls}">
        <div class="sensor-name">${label}</div>
        <div class="sensor-val">${s.val.toLocaleString()}<span class="sensor-unit">${s.unit}</span></div>
        <div class="sensor-bar">
          <div class="sensor-bar-fill" style="width:${pct}%;background:${barColor};"></div>
        </div>
      </div>`;
  });
}

// ── Class probability bars ─────────────────────────────────
function renderClassProbs(probs, containerId){
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  Object.entries(probs).forEach(([cls, p])=>{
    const col = CLASS_COLORS[cls]||'#00d4ff';
    el.innerHTML += `
      <div class="class-row">
        <div class="class-name">${cls}</div>
        <div class="class-bar-bg">
          <div class="class-bar-fill" style="width:${(p*100).toFixed(1)}%;background:${col};"></div>
        </div>
        <div class="class-pct">${(p*100).toFixed(1)}%</div>
      </div>`;
  });
}

// ── Probability trend chart ────────────────────────────────
function buildProbChart(history, canvasId, multi){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');

  const labels = history.map((_,i)=>`T-${history.length-1-i}`).reverse();

  if(multi){
    // All motors
    const datasets = Object.entries(MOTORS).map(([id,m])=>({
      label: id,
      data: m.history,
      borderColor: id==='MOTOR-01'?'#00d4ff':id==='MOTOR-02'?'#ffaa00':'#ff2244',
      backgroundColor:'transparent',
      borderWidth:2,
      pointRadius:3,
      tension:0.4,
    }));
    if(allMotorsChart) allMotorsChart.destroy();
    allMotorsChart = new Chart(ctx,{
      type:'line',
      data:{labels:labels,datasets},
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{legend:{labels:{color:'#6a8fa8',font:{family:'JetBrains Mono',size:10}}}},
        scales:{
          x:{ticks:{color:'#334455',font:{family:'JetBrains Mono',size:9}},grid:{color:'#0f1e30'}},
          y:{min:0,max:100,ticks:{color:'#334455',font:{family:'JetBrains Mono',size:9},callback:v=>v+'%'},grid:{color:'#0f1e30'}},
        }
      }
    });
    return;
  }

  const colors = history.map(v=>v>=60?'#ff2244':v>=25?'#ffaa00':'#00ff88');
  if(probChart) probChart.destroy();
  probChart = new Chart(ctx,{
    type:'line',
    data:{
      labels:labels,
      datasets:[{
        label:'Failure prob %',
        data:history,
        borderColor:'#00d4ff',
        backgroundColor:'rgba(0,212,255,0.05)',
        borderWidth:2,
        pointBackgroundColor:colors,
        pointRadius:5,
        tension:0.4,
        fill:true,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        annotation:{annotations:{
          high:{type:'line',yMin:60,yMax:60,borderColor:'rgba(255,34,68,0.4)',borderWidth:1,borderDash:[4,4]},
          med:{type:'line',yMin:25,yMax:25,borderColor:'rgba(255,170,0,0.4)',borderWidth:1,borderDash:[4,4]},
        }}
      },
      scales:{
        x:{ticks:{color:'#334455',font:{family:'JetBrains Mono',size:9}},grid:{color:'#0f1e30'}},
        y:{min:0,max:100,ticks:{color:'#334455',font:{family:'JetBrains Mono',size:9},callback:v=>v+'%'},grid:{color:'#0f1e30'}},
      }
    }
  });
}

// ── Main dashboard render ──────────────────────────────────
function renderDashboard(){
  const m = MOTORS[currentMotor];
  document.getElementById('selected-motor-badge').textContent = currentMotor;
  document.getElementById('last-updated').textContent = 'Last updated: '+new Date().toTimeString().slice(0,8);

  // KPIs
  document.getElementById('kpi-prob').innerHTML = (m.prob*100).toFixed(1)+'<span style="font-size:16px;">%</span>';
  document.getElementById('kpi-health').innerHTML = m.health.toFixed(1)+'<span style="font-size:16px;">/100</span>';
  document.getElementById('kpi-mode').textContent = m.mode;
  const probConf = m.probs[m.mode]||0;
  document.getElementById('kpi-mode-conf').textContent = 'Confidence '+(probConf*100).toFixed(1)+'%';

  // Alerts
  const crit = Object.entries(m.sensors).filter(([k,s])=>s.val>s.warn);
  document.getElementById('kpi-alerts').textContent = crit.length;
  document.getElementById('kpi-alerts-sub').textContent =
    crit.length ? crit.map(([k])=>k.replace(/_/g,' ')).join(', ') : 'All parameters nominal';

  const alertColor = m.prob>=0.6?'red':m.prob>=0.25?'amber':'green';
  const probCard = document.querySelector('.kpi-card.cyan');
  probCard.style.setProperty('--accent', 'var(--'+alertColor+')');

  // Gauge & ring
  updateGauge(m.prob);
  updateHealthRing(m.health);

  // Class probs
  renderClassProbs(m.probs, 'class-probs-panel');

  // Sensor grid
  renderSensorGrid(currentMotor);

  // Prob trend
  setTimeout(()=>buildProbChart(m.history,'prob-chart',false), 50);
}

// ── Input table builder ────────────────────────────────────
const DEFAULTS = {
  Winding_Temp_K:328, Bearing_Temp_K:322, Rotational_Speed_RPM:1448,
  Torque_Nm:36, Vibration_mmps:1.8, Current_Imbalance_pct:1.5,
  Insulation_Resistance_MOhm:210
};
const SENSOR_KEYS = Object.keys(DEFAULTS);
const SENSOR_LABELS = {
  Winding_Temp_K:'Winding Temp (K)',
  Bearing_Temp_K:'Bearing Temp (K)',
  Rotational_Speed_RPM:'RPM',
  Torque_Nm:'Torque (Nm)',
  Vibration_mmps:'Vibration (mm/s)',
  Current_Imbalance_pct:'Curr. Imbalance (%)',
  Insulation_Resistance_MOhm:'Insulation (MΩ)',
};

function buildInputTable(){
  const n = parseInt(document.getElementById('n-readings').value)||6;
  const t = document.getElementById('input-table');
  let html = '<thead><tr><th>#</th>';
  SENSOR_KEYS.forEach(k=>{ html+=`<th>${SENSOR_LABELS[k]}</th>`; });
  html += '</tr></thead><tbody>';
  for(let i=0;i<n;i++){
    html += `<tr><td style="color:var(--text2);">${i===n-1?'★ Latest':i+1}</td>`;
    SENSOR_KEYS.forEach(k=>{
      html += `<td><input type="number" step="0.1" value="${DEFAULTS[k]}" id="inp_${i}_${k}"></td>`;
    });
    html += '</tr>';
  }
  html += '</tbody>';
  t.innerHTML = html;
}

function fillDefaults(){
  const n = parseInt(document.getElementById('n-readings').value)||6;
  for(let i=0;i<n;i++){
    SENSOR_KEYS.forEach(k=>{
      const el = document.getElementById(`inp_${i}_${k}`);
      if(el) el.value = DEFAULTS[k];
    });
  }
}

// ── Simulated prediction ───────────────────────────────────
function runPrediction(){
  const n = parseInt(document.getElementById('n-readings').value)||6;
  const motorId = document.getElementById('pred-motor-id').value;

  // Read input values
  const readings = [];
  for(let i=0;i<n;i++){
    const row = {};
    SENSOR_KEYS.forEach(k=>{
      const el = document.getElementById(`inp_${i}_${k}`);
      row[k] = parseFloat(el?el.value:DEFAULTS[k]);
    });
    readings.push(row);
  }

  // Compute rolling features on client side (simplified)
  const last = readings[readings.length-1];
  const means = {};
  SENSOR_KEYS.forEach(k=>{
    means[k] = readings.reduce((a,r)=>a+r[k],0)/readings.length;
  });
  const deltas = {};
  SENSOR_KEYS.forEach(k=>{ deltas[k] = last[k]-means[k]; });

  // Heuristic scoring (demo — in production this calls the Python model)
  let score = 0;
  if(last.Winding_Temp_K > 370) score += 0.3;
  if(last.Vibration_mmps > 6)   score += 0.35;
  if(last.Current_Imbalance_pct > 5) score += 0.2;
  if(last.Rotational_Speed_RPM < 1200) score += 0.25;
  if(deltas.Winding_Temp_K > 8) score += 0.15;
  if(deltas.Vibration_mmps > 1) score += 0.15;
  score = Math.min(score, 0.98);

  // Determine mode
  let mode = 'Normal';
  if(last.Vibration_mmps > 7.1) mode = 'VBF - Vibration';
  else if(last.Winding_Temp_K > 380) mode = 'HDF - Heat';
  else if(last.Current_Imbalance_pct > 5) mode = 'PWF - Electrical';
  else if(last.Rotational_Speed_RPM < 1100) mode = 'OSF - Overspeed';
  else if(score > 0.35) mode = 'TWF - Wear';

  const health = Math.max(0, 100 - score*100);
  const conf = 0.7 + Math.random()*0.25;

  // Build class probs
  const probs = {};
  OBJECT_KEYS(['Normal','HDF - Heat','OSF - Overspeed','TWF - Wear','PWF - Electrical','VBF - Vibration']).forEach((c,i)=>{
    probs[c] = c===mode ? conf : (1-conf)/5;
  });
  if(mode==='Normal') probs['Normal'] = 1-score;

  // Show results
  const rc = document.getElementById('res-prob-card');
  const col = probToColor(score);
  rc.querySelector('.kpi-value').style.color = col;
  document.getElementById('res-prob-val').innerHTML = (score*100).toFixed(1)+'<span style="font-size:16px;">%</span>';
  document.getElementById('res-mode-val').textContent = mode;
  document.getElementById('res-health-val').innerHTML = health.toFixed(1)+'<span style="font-size:16px;">/100</span>';
  document.getElementById('res-conf-val').innerHTML = (conf*100).toFixed(1)+'<span style="font-size:16px;">%</span>';
  document.getElementById('res-motor-label').textContent = motorId;

  // Action
  const ap = document.getElementById('res-action-panel');
  const isAlert = score>=0.6;
  const bgCol = isAlert?'rgba(255,34,68,0.06)':score>=0.25?'rgba(255,170,0,0.06)':'rgba(0,255,136,0.06)';
  const bdCol = isAlert?'rgba(255,34,68,0.25)':score>=0.25?'rgba(255,170,0,0.25)':'rgba(0,255,136,0.25)';
  const txCol = isAlert?'var(--red)':score>=0.25?'var(--amber)':'var(--green)';
  ap.innerHTML = `<div style="background:${bgCol};border:1px solid ${bdCol};border-radius:8px;padding:14px 16px;font-size:13px;color:${txCol};line-height:1.7;">${ACTIONS[mode]}</div>`;

  // Class probs
  renderClassProbs(probs, 'res-class-panel');

  // Trend insight
  const trendPanel = document.getElementById('trend-charts');
  trendPanel.innerHTML = '';
  ['Winding_Temp_K','Vibration_mmps'].forEach(key=>{
    const vals = readings.map(r=>r[key]);
    const delta = vals[vals.length-1]-vals[0];
    const sign = delta>0?'+':'';
    const dColor = delta>0?(key==='Insulation_Resistance_MOhm'?'var(--green)':'var(--red)'):'var(--green)';
    trendPanel.innerHTML += `
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title" style="font-size:11px;">${SENSOR_LABELS[key]}</div>
          <span style="font-family:var(--font-mono);font-size:10px;color:${dColor};margin-left:auto;">${sign}${delta.toFixed(2)} over window</span>
        </div>
        <div class="panel-body" style="height:100px;display:flex;align-items:flex-end;gap:4px;">
          ${vals.map((v,i)=>{
            const mn=Math.min(...vals), mx=Math.max(...vals);
            const h = mx===mn?50:((v-mn)/(mx-mn)*70+10);
            const c = i===vals.length-1?'var(--cyan)':'var(--border)';
            return `<div style="flex:1;height:${h}px;background:${c};border-radius:2px 2px 0 0;transition:height 0.5s;"></div>`;
          }).join('')}
        </div>
      </div>`;
  });

  document.getElementById('pred-results').style.display='block';
  document.getElementById('pred-results').scrollIntoView({behavior:'smooth',block:'start'});
}

function OBJECT_KEYS(arr){return arr;}

// ── Trend page ─────────────────────────────────────────────
function buildTrendPage(){
  setTimeout(()=>buildProbChart([],'all-motors-chart',true),50);

  const panels = document.getElementById('sensor-trend-panels');
  panels.innerHTML = '';
  const sensors = ['Winding_Temp_K','Vibration_mmps','Torque_Nm','Bearing_Temp_K','Rotational_Speed_RPM','Current_Imbalance_pct'];
  const canvasIds = [];

  sensors.forEach((s,idx)=>{
    const cid = 'trend-canvas-'+idx;
    canvasIds.push(cid);
    panels.innerHTML += `
      <div class="panel">
        <div class="panel-header"><div class="panel-title" style="font-size:11px;">${SENSOR_LABELS[s]||s}</div></div>
        <div class="panel-body"><div class="chart-canvas-wrap"><canvas class="trend" id="${cid}"></canvas></div></div>
      </div>`;
  });

  setTimeout(()=>{
    sensors.forEach((s,idx)=>{
      const canvas = document.getElementById('trend-canvas-'+idx);
      if(!canvas) return;
      const ctx = canvas.getContext('2d');
      const datasets = Object.entries(MOTORS).map(([id,m])=>({
        label:id,
        data: Array.from({length:8},(_,i)=>{
          const base = m.sensors[s]?.val||0;
          return +(base + (Math.random()-0.5)*base*0.05).toFixed(2);
        }),
        borderColor:id==='MOTOR-01'?'#00d4ff':id==='MOTOR-02'?'#ffaa00':'#ff2244',
        backgroundColor:'transparent',
        borderWidth:1.5,
        pointRadius:2,
        tension:0.4,
      }));
      if(trendCharts[idx]) trendCharts[idx].destroy();
      trendCharts[idx] = new Chart(ctx,{
        type:'line',
        data:{labels:['T-7','T-6','T-5','T-4','T-3','T-2','T-1','Now'],datasets},
        options:{
          responsive:true,maintainAspectRatio:false,
          plugins:{legend:{labels:{color:'#334455',font:{size:9,family:'JetBrains Mono'},boxWidth:10}}},
          scales:{
            x:{ticks:{color:'#334455',font:{size:8,family:'JetBrains Mono'}},grid:{color:'#0a1520'}},
            y:{ticks:{color:'#334455',font:{size:8,family:'JetBrains Mono'}},grid:{color:'#0a1520'}},
          }
        }
      });
    });
  }, 100);
}

// ── Model page ─────────────────────────────────────────────
function buildModelPage(){
  const classes = [
    {name:'Normal',       desc:'All parameters within rated limits',                color:'#00ff88', f1:'0.98', prec:'0.97', rec:'1.00'},
    {name:'HDF - Heat',   desc:'Thermal overload / cooling failure',                color:'#ff2244', f1:'0.60', prec:'0.75', rec:'0.50'},
    {name:'OSF - Overspeed',desc:'Speed below rated under load',                    color:'#ffaa00', f1:'0.80', prec:'1.00', rec:'0.67'},
    {name:'TWF - Wear',   desc:'Tool / brush / bearing wear accumulation',          color:'#ff7700', f1:'—',    prec:'—',    rec:'—'},
    {name:'PWF - Electrical',desc:'Phase imbalance / insulation degradation',       color:'#aa44ff', f1:'0.80', prec:'1.00', rec:'0.67'},
    {name:'VBF - Vibration',desc:'Mechanical imbalance / misalignment',             color:'#00d4ff', f1:'1.00', prec:'1.00', rec:'1.00'},
  ];

  const list = document.getElementById('class-detail-list');
  list.innerHTML = classes.map(c=>`
    <div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--border2);">
      <div style="width:10px;height:10px;border-radius:50%;background:${c.color};flex-shrink:0;margin-top:3px;box-shadow:0 0 6px ${c.color};"></div>
      <div>
        <div style="font-family:var(--font-mono);font-size:11px;color:${c.color};font-weight:500;">${c.name}</div>
        <div style="font-size:12px;color:var(--text2);">${c.desc}</div>
      </div>
    </div>`).join('');

  const perf = document.getElementById('perf-table');
  perf.innerHTML = `
    <table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:11px;">
      <thead>
        <tr>
          ${['Class','Precision','Recall','F1-Score'].map(h=>`<th style="text-align:left;padding:8px 12px;border-bottom:1px solid var(--border2);color:var(--text2);letter-spacing:0.08em;font-size:9px;text-transform:uppercase;">${h}</th>`).join('')}
        </tr>
      </thead>
      <tbody>
        ${classes.map((c,i)=>`
          <tr style="background:${i%2?'transparent':'rgba(255,255,255,0.01)'};">
            <td style="padding:8px 12px;color:${c.color};">${c.name}</td>
            <td style="padding:8px 12px;color:var(--text);">${c.prec}</td>
            <td style="padding:8px 12px;color:var(--text);">${c.rec}</td>
            <td style="padding:8px 12px;color:var(--text);">${c.f1}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

// ── Init ───────────────────────────────────────────────────
buildInputTable();
renderDashboard();

// Simulate live data drift
setInterval(()=>{
  Object.values(MOTORS).forEach(m=>{
    m.history.push(+(m.prob*100 + (Math.random()-0.4)*3).toFixed(1));
    if(m.history.length>20) m.history.shift();
    m.prob = Math.min(0.98, Math.max(0.01, m.prob + (Math.random()-0.48)*0.01));
    m.health = Math.max(0, 100 - m.prob*100);
  });
  document.getElementById('last-updated').textContent = 'Last updated: '+new Date().toTimeString().slice(0,8);
  if(document.getElementById('page-dashboard').classList.contains('active')){
    renderDashboard();
  }
}, 4000);
</script>
</body>
</html>
