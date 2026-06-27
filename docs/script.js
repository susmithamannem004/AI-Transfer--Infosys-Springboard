const header = document.querySelector(".site-header");
const links = document.querySelectorAll(".nav a");

const setHeaderShadow = () => {
  header.style.boxShadow = window.scrollY > 10
    ? "0 10px 34px rgba(35, 45, 40, .08)"
    : "none";
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    links.forEach((link) => {
      link.toggleAttribute("aria-current", link.getAttribute("href") === `#${entry.target.id}`);
    });
  });
}, { rootMargin: "-40% 0px -55% 0px" });

document.querySelectorAll("section[id]").forEach((section) => observer.observe(section));
window.addEventListener("scroll", setHeaderShadow, { passive: true });
setHeaderShadow();
