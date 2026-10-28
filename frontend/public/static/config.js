// Configuración del frontend. La API se consulta en la misma dirección (/api): en local la
// redirige nginx al contenedor del backend y en la nube la sirve el mismo contenedor.
// Los tokens son de DEMOSTRACIÓN (simulan el inicio de sesión de los 5 clientes del piloto).
window.CAB_CONFIG = {
  apiBase: "",
  clientes: [
    { token: "demo-arm01", nombre: "ARM-01 · Armadora (Silao)" },
    { token: "demo-arm03", nombre: "ARM-03 · Armadora" },
    { token: "demo-arm04", nombre: "ARM-04 · Armadora" },
    { token: "demo-arm07", nombre: "ARM-07 · Armadora (EE. UU.)" },
    { token: "demo-tier02", nombre: "TIER-02 · Proveedor Tier 1" }
  ],
  ejemplos: [
    "¿Cómo va mi pedido?",
    "Quiero hablar con mi ejecutivo",
    "Llegaron piezas dañadas, quiero poner una queja",
    "Necesito la factura de octubre",
    "When will my order arrive?"
  ]
};
