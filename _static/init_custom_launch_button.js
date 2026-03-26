document.addEventListener("DOMContentLoaded", function() {
    const dropdownButton = document.querySelector(".dropdown-source-buttons");
    const dropdownLinks = dropdownButton.querySelectorAll(".dropdown-menu .dropdown-item");

    // set click event for update url
    dropdownLinks.forEach(link => {
        link.addEventListener("click", function (event) {
            const baseUrl = decodeURIComponent(link.getAttribute("href"));
            const url = getPath(baseUrl);
            if (url) {
                event.preventDefault();
                window.open(url, '_blank');
            }
        }, true);
    });

    // Set tooltip
    dropdownLinks.forEach(link => {
        let url = decodeURIComponent(link.getAttribute("href"));
        if (url.includes("mybinder")) {
            const tooltip = `
              <b>Launch on Binder</b>
              <img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder" height="30">
            `
            link.setAttribute("data-bs-title", tooltip);
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
        dropdownButton.style.display = "";
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
        dropdownButton.style.display = 'none';
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
