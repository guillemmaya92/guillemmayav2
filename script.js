document.addEventListener("DOMContentLoaded", function () {
  const darkModeToggle = document.querySelector('[data-bs-toggle="theme"]');

  if (darkModeToggle) {
    // Actualiza el icono al cargar la página
    updateIcon();

    // Cambia el icono al hacer clic
    darkModeToggle.addEventListener("click", updateIcon);
  }

  function updateIcon() {
    const isDark = document.documentElement.getAttribute("data-bs-theme") === "dark";
    darkModeToggle.innerHTML = isDark ? "☀️" : "🌙";
  }
});