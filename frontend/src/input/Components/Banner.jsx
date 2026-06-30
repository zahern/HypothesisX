export default function Banner({ appData, setAppData, theme, setTheme }) {
  return (
    <div class="banner">
      <div class="banner_logo">λ</div>
      <div>
        <div class="banner_title">DCM Studio</div>
        <div class="banner_subtitle">DISCRETE CHOICE MODELLING</div>
      </div>
      <div class="flex_spacer" />

      {/* Dark/Light theme switching */}
      <div class="banner_mode_group">
        <div key="dark" onClick={() => setTheme("dark")}
          class={`mode_button ${theme==="dark" ? "mode_button_select" : ""}`}>Dark</div>
        <div key="light" onClick={() => setTheme("light")}
          class={`mode_button ${theme==="light" ? "mode_button_select" : ""}`}>Light</div>
      </div>
    </div>
  )
}
