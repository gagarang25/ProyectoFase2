// Frontend del chatbot de CAB: consume la API REST del backend con fetch().
(function () {
  "use strict";
  const C = window.CAB_CONFIG;
  const $ = (id) => document.getElementById(id);
  const estado = { token: C.clientes[0].token, ocContexto: null, ultimoMensaje: null, ocupado: false };

  const ACCIONES = {
    automatica: "Respuesta automática",
    aclarar: "Pregunta de aclaración",
    escalar: "Escalado a ejecutivo",
    canalizar: "Fuera de alcance",
    pedir_dato: "Pide el número de OC",
    oc_no_encontrada: "OC no encontrada"
  };

  const ESTATUS = { en_produccion: "en producción", lista_para_embarque: "lista para embarque",
    en_transito: "en tránsito", entregada: "entregada" };

  async function api(ruta, opciones = {}) {
    const resp = await fetch(C.apiBase + ruta, {
      ...opciones,
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + estado.token, ...(opciones.headers || {}) }
    });
    const cuerpo = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detalle = Array.isArray(cuerpo.detail) ? cuerpo.detail.map((d) => d.msg).join("; ") : cuerpo.detail;
      throw new Error(`${resp.status} · ${detalle || resp.statusText}`);
    }
    return { cuerpo, ms: resp.headers.get("X-Tiempo-ms") };
  }

  function burbuja(texto, quien, extra) {
    const div = document.createElement("div");
    div.className = "burbuja " + quien;
    div.textContent = texto;
    if (extra) {
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = extra;
      div.appendChild(meta);
    }
    $("mensajes").appendChild(div);
    $("mensajes").scrollTop = $("mensajes").scrollHeight;
  }

  function mostrarOpciones(opciones) {
    const cont = $("opciones");
    cont.innerHTML = "";
    (opciones || []).forEach((op) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = op.texto;
      b.onclick = () => enviar(estado.ultimoMensaje || op.texto, op.intencion, op.texto);
      cont.appendChild(b);
    });
  }

  function mostrarDetalle(r, ms) {
    const filas = [
      ["Intención", `${r.intencion} (${(r.confianza * 100).toFixed(1)} %)`],
      ["Acción", ACCIONES[r.accion] || r.accion],
      ["Idioma", r.idioma],
      ["OC", r.oc || "—"],
      ["Requiere ejecutivo", r.requiere_ejecutivo ? "sí" + (r.folio ? " · " + r.folio : "") : "no"],
      ["Tiempo total API", ms ? ms + " ms" : "—"]
    ];
    const est = r.datos && r.datos.estimacion;
    if (est) {
      filas.push(["ETA (días)", `${est.eta_dias} [${est.eta_min_dias}–${est.eta_max_dias}]`]);
      filas.push(["Prob. de retraso", (est.prob_retraso * 100).toFixed(1) + " %" + (est.riesgo_alto ? " · alto" : "")]);
    }
    let html = "<table>" + filas.map(([k, v]) => `<tr><th>${k}</th><td>${escapar(String(v))}</td></tr>`).join("") + "</table>";
    html += "<ol class='pasos'>" + (r.traza || []).map((p) =>
      `<li><strong>${escapar(p.paso)}</strong> ${escapar(p.detalle)} <span>${p.ms} ms</span></li>`).join("") + "</ol>";
    html += `<details><summary>JSON completo</summary><pre>${escapar(JSON.stringify(r, null, 2))}</pre></details>`;
    $("detalle").innerHTML = html;
  }

  function escapar(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  async function enviar(mensaje, intencionElegida, textoVisible) {
    if (estado.ocupado || !mensaje.trim()) return;
    estado.ocupado = true;
    $("enviar").disabled = true;
    mostrarOpciones([]);
    burbuja(textoVisible || mensaje, "cliente");
    if (!intencionElegida) estado.ultimoMensaje = mensaje;
    try {
      const { cuerpo, ms } = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ mensaje, oc_contexto: estado.ocContexto, intencion_elegida: intencionElegida || null })
      });
      if (cuerpo.oc) estado.ocContexto = cuerpo.oc;
      burbuja(cuerpo.respuesta, "bot", ACCIONES[cuerpo.accion] || cuerpo.accion);
      mostrarOpciones(cuerpo.opciones);
      mostrarDetalle(cuerpo, ms);
    } catch (e) {
      burbuja("No pude comunicarme con la API: " + e.message, "error");
    } finally {
      estado.ocupado = false;
      $("enviar").disabled = false;
      $("texto").focus();
    }
  }

  async function cargarOrdenes() {
    const cont = $("ordenes");
    cont.innerHTML = "<span class='nota'>Cargando…</span>";
    try {
      const { cuerpo } = await api("/api/ordenes?solo_abiertas=true");
      cont.innerHTML = cuerpo.ordenes.length ? "" : "<span class='nota'>Sin órdenes abiertas.</span>";
      cuerpo.ordenes.slice(0, 8).forEach((o) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip";
        b.innerHTML = `<strong>${o.oc}</strong><span>${ESTATUS[o.estatus] || o.estatus}</span>`;
        b.onclick = () => enviar(`¿Cuándo llega la OC ${o.oc}?`);
        cont.appendChild(b);
      });
    } catch (e) {
      cont.innerHTML = `<span class='nota error'>${escapar(e.message)}</span>`;
    }
  }

  async function revisarSalud() {
    const p = $("salud");
    try {
      const resp = await fetch(C.apiBase + "/api/salud");
      const s = await resp.json();
      p.textContent = s.estado === "ok" ? `API ok · router ${s.router.version || ""}` : "API degradada: router no disponible";
      p.className = "pildora " + (s.estado === "ok" ? "ok" : "alerta");
      p.title = `Commit ${s.git_commit} · SAP simulado al ${s.fecha_referencia_sap}`;
    } catch (e) {
      p.textContent = "API sin conexión";
      p.className = "pildora alerta";
    }
  }

  function iniciar() {
    const sel = $("cliente");
    C.clientes.forEach((c) => sel.add(new Option(c.nombre, c.token)));
    sel.onchange = () => {
      estado.token = sel.value;
      estado.ocContexto = null;
      $("mensajes").innerHTML = "";
      mostrarOpciones([]);
      saludo();
      cargarOrdenes();
    };
    C.ejemplos.forEach((t) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = t;
      b.onclick = () => enviar(t);
      $("ejemplos").appendChild(b);
    });
    $("formulario").onsubmit = (ev) => {
      ev.preventDefault();
      const t = $("texto").value;
      $("texto").value = "";
      enviar(t);
    };
    saludo();
    revisarSalud();
    cargarOrdenes();
  }

  function saludo() {
    burbuja("Hola, soy el asistente de órdenes de CAB. Puedo decirte el estatus de una orden de compra o su fecha estimada de entrega. Para otros temas te comunico con tu ejecutivo.", "bot");
  }

  iniciar();
})();
