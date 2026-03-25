document.addEventListener("click", function (event) {
    const link = event.target.closest(".dropdown-source-buttons .dropdown-menu .dropdown-item");
    if (!link) return;

    if (window.location.pathname.includes("methods")) {
        const baseUrl = decodeURIComponent(link.getAttribute("href"));
        const url = getPath(baseUrl);
        if (url) {
            event.preventDefault();
            window.open(url, '_blank');
        }
    }
}, true);

document.addEventListener("DOMContentLoaded", function() {
    const dropdownLinks = document.querySelector(".dropdown-source-buttons").querySelectorAll(".dropdown-menu .dropdown-item");

    // Set tooltip
    dropdownLinks.forEach(link => {
        let url = decodeURIComponent(link.getAttribute("href"));
        if (url.includes("mybinder")) {
            link.setAttribute("data-bs-title", "<b>Launch on Binder</b>");
        } else if (url.includes("colab")) {
            link.setAttribute("data-bs-title", "<b>Launch on Colab</b>");
        } else if (url.includes("sagemaker")) {
            link.setAttribute("data-bs-title", "<b>Launch on Sagemaker Studio Lab</b>");
        }
        new bootstrap.Tooltip(link, {
            html: true,
            placement: 'left'
        });
    });

    // hide launch button in main page
    // hide launch options with metadata launch_on
    const path = window.location.pathname
    if (path.includes("methods")) {
        document.querySelector(".dropdown-source-buttons").style.display = "";
        const metadataLaunch = document.querySelector('meta[name="launch_on"]')
        const launchOn = metadataLaunch ? metadataLaunch.content.split(",") : [];
        dropdownLinks.forEach(link => {
            const href = link.getAttribute("href");
            if (launchOn.some(item => href.includes(item)) || launchOn.length === 0) {
                link.style.display = "";
            } else {
                link.style.display = "none";
            }
        });
    } else {
        document.querySelector(".dropdown-source-buttons").style.display = 'none';
    }
});

function getPath(baseUrl) {
    const path = window.location.pathname
    const currentPath = path.split('/').slice(-2);
    const branch = document.querySelector('meta[name="launch_branch"]').content;
    if (baseUrl.includes("mybinder")) {
        return baseUrl + currentPath[0] + "/" + branch + "?filepath=" + currentPath[1].replace('.html', '.ipynb');
    } else if (baseUrl.includes("colab")) {
        return baseUrl + currentPath[0] + "/blob/" + branch + "/" + currentPath[1].replace('.html', '.ipynb');
    } else if (baseUrl.includes("sagemaker")) {
        return baseUrl + currentPath[0] + "/blob/" + branch + "/" + currentPath[1].replace('.html', '.ipynb');
    }

    return "";
}
