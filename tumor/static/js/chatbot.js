document.addEventListener("DOMContentLoaded", function () {
  const container = document.getElementById("chatbot-container");
  if (!container) return; // login/signup sahifalarida bo'lmasligi mumkin

  const button = document.getElementById("chatbot-button");
  const windowBox = document.getElementById("chatbot-window");
  const closeBtn = document.getElementById("chatbot-close");
  const form = document.getElementById("chatbot-form");
  const input = document.getElementById("chatbot-input");
  const messages = document.getElementById("chatbot-messages");

  const chatUrl = container.dataset.chatUrl || "/api/chat/";

  // 1. Drag-and-drop uchun o'zgaruvchilar
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let startLeft = 0;
  let startTop = 0;
  let moved = false;

  // Drag boshlash
  button.addEventListener("mousedown", function (e) {
    isDragging = true;
    moved = false;

    startX = e.clientX;
    startY = e.clientY;

    const rect = container.getBoundingClientRect();
    startLeft = rect.left;
    startTop = rect.top;

    document.body.style.userSelect = "none";
  });

  // Harakat
  document.addEventListener("mousemove", function (e) {
    if (!isDragging) return;

    const dx = e.clientX - startX;
    const dy = e.clientY - startY;

    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      moved = true;
    }

    let newLeft = startLeft + dx;
    let newTop = startTop + dy;

    const maxLeft = window.innerWidth - container.offsetWidth;
    const maxTop = window.innerHeight - container.offsetHeight;

    newLeft = Math.max(0, Math.min(newLeft, maxLeft));
    newTop = Math.max(0, Math.min(newTop, maxTop));

    container.style.left = newLeft + "px";
    container.style.top = newTop + "px";
    container.style.right = "auto";
    container.style.bottom = "auto";

    // Oyna ochiq bo'lsa - joylashuvni doim yangilab turamiz
    if (windowBox.style.display === "block") {
      updateWindowPosition();
    }
  });

  // Drag tugatish
  document.addEventListener("mouseup", function () {
    if (!isDragging) return;

    isDragging = false;
    document.body.style.userSelect = "";

    // Agar deyarli siljimagan bo'lsa - bu klik
    if (!moved) {
      toggleChatWindow();
    }
  });

  // 2. Chat oynasini iconka joylashuviga qarab joylashtirish
  function updateWindowPosition() {
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    const buttonRect = button.getBoundingClientRect();

    let wasHidden = false;
    if (windowBox.style.display !== "block") {
      wasHidden = true;
      windowBox.style.visibility = "hidden";
      windowBox.style.display = "block";
    }

    const windowHeight = windowBox.offsetHeight || 440;
    const windowWidth = windowBox.offsetWidth || 340;

    if (wasHidden) {
      windowBox.style.display = "none";
      windowBox.style.visibility = "";
    }

    const spaceAbove = buttonRect.top;
    const spaceBelow = viewportHeight - buttonRect.bottom;
    const centerX = buttonRect.left + buttonRect.width / 2;

    // Vertikal: tepaga yoki pastga
    if (spaceBelow >= windowHeight + 20) {
      windowBox.style.top = buttonRect.height + 10 + "px";
    } else if (spaceAbove >= windowHeight + 20) {
      windowBox.style.top = -(windowHeight + 10) + "px";
    } else {
      if (spaceBelow >= spaceAbove) {
        windowBox.style.top = Math.max(buttonRect.height + 10, 10) + "px";
      } else {
        windowBox.style.top =
          -Math.max(Math.min(windowHeight, spaceAbove - 10), 60) + "px";
      }
    }

    // Gorizontal: chap yoki o'ng
    if (centerX > viewportWidth / 2) {
      windowBox.style.right = "0";
      windowBox.style.left = "";
    } else {
      windowBox.style.left = "0";
      windowBox.style.right = "";
    }
  }

  // 3. Chat oynasini ochish yopish
  function toggleChatWindow() {
    if (windowBox.style.display === "block") {
      windowBox.style.display = "none";
    } else {
      windowBox.style.display = "block";
      updateWindowPosition();
      input.focus();
    }
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", function () {
      windowBox.style.display = "none";
    });
  }

  // 4. Xabar qo'shish yordamchi funksiyasi
  function addMessage(text, role) {
    const div = document.createElement("div");
    div.classList.add("chat-message", role);
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  // 5. Chat formasini yuborish - backendga so'rov
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    const loadingId = "bot-loading";
    const loadingDiv = document.createElement("div");
    loadingDiv.id = loadingId;
    loadingDiv.classList.add("chat-message", "bot");
    loadingDiv.textContent = "Yuklanmoqda...";
    messages.appendChild(loadingDiv);
    messages.scrollTop = messages.scrollHeight;

    fetch(chatUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
        history: [],
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        const ld = document.getElementById(loadingId);
        if (ld) ld.remove();

        const reply = data.reply || "Serverdan javob kelmadi.";
        addMessage(reply, "bot");
      })
      .catch((err) => {
        console.error(err);
        const ld = document.getElementById(loadingId);
        if (ld) ld.remove();
        addMessage(
          "Xatolik yuz berdi. Keyinroq yana urinib ko‘ring.",
          "bot"
        );
      });
  });
});
