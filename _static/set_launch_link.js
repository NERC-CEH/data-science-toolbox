document.addEventListener("click", function (event) {
    const link = event.target.closest(".dropdown-menu .dropdown-item");
    if (!link) return;

    event.preventDefault();
    const path = window.location.pathname
    if (path.includes("methods")) {
        const currentPath = path.split('/').slice(-2);
        const branch = document.querySelector('meta[name="launch_branch"]').content;
        let url = decodeURIComponent(link.getAttribute("href"));
        if (url.includes("mybinder")) {
            url = url + currentPath[0] + "/" + branch + "?filepath=" + currentPath[1].replace('.html', '.ipynb');
            window.open(url, '_blank');
        } else if (url.includes("colab")) {
            url = url + currentPath[0] + "/blob/" + branch + "/" + currentPath[1].replace('.html', '.ipynb');
            window.open(url, '_blank');
        } else if (url.includes("sagemaker")) {
            url = url + currentPath[0] + "/blob/" + branch + "/" + currentPath[1].replace('.html', '.ipynb');
            window.open(url, '_blank');
        }
    } else {
      alert("Please navigate to notebook page to launch.")
    }
}, true);
