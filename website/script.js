const visibleVideos = new Set();

function playVisibleVideo(video) {
  if (visibleVideos.has(video) && !document.hidden) {
    video.muted = true;
    video.play().catch(() => {});
  }
}

document.querySelectorAll('.view-switch').forEach(group => {
  group.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button || button.getAttribute('aria-pressed') === 'true') return;
    const video = group.closest('figure').querySelector('video');
    const time = video.currentTime;
    video.pause();
    group.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    video.poster = button.dataset.poster;
    video.querySelector('source').src = button.dataset.src;
    video.addEventListener('loadedmetadata', () => {
      video.currentTime = Math.min(time, Math.max(0, video.duration - 0.05));
      playVisibleVideo(video);
    }, {once:true});
    video.load();
  });
});
const videoObserver = new IntersectionObserver(entries => {
  entries.forEach(({target: video, isIntersecting}) => {
    if (isIntersecting) {
      visibleVideos.add(video);
      playVisibleVideo(video);
    } else {
      visibleVideos.delete(video);
      video.pause();
    }
  });
}, {threshold: 0});

document.querySelectorAll('video').forEach(video => {
  video.muted = true;
  video.playsInline = true;
  videoObserver.observe(video);
});

document.addEventListener('visibilitychange', () => {
  visibleVideos.forEach(video => {
    if (document.hidden) video.pause();
    else playVisibleVideo(video);
  });
});
