export default function Banner({appData, setAppData}) {
  return (
    <div class="banner">
      <a href="/" class="banner_home">
        <div class="banner_logo">λ</div>
        <div>
          <div class="banner_title">DCM Studio</div>
          <div class="banner_subtitle">DISCRETE CHOICE MODELLING</div>
        </div>
      </a>
      <div class="flex_spacer" />

      {/* Dark/Light theme switching */}
      <div class="banner_mode_group">
        <div key="dark" onClick={() => setAppData({...appData, theme:"dark"})}
          class={`mode_button ${appData.theme==="dark" ? "mode_button_select" : ""}`}>Dark</div>
        <div key="light" onClick={() => setAppData({...appData, theme:"light"})}
          class={`mode_button ${appData.theme==="light" ? "mode_button_select" : ""}`}>Light</div>
      </div>
    </div>
  )
}

