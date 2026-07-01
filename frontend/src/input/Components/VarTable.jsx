import "../All.css"

/*
  This file was attempting to make the VariableOptions components
  in both the Estimation and Search -Sub-Components files smaller
  it didnt really help with that and just made the implimentation
  more confusing
*/

export function VarTable({ children, legend, summary }) {
  /*
    Outer shell for the per-variable options table. Caller supplies the
    <VarTableHeader/> and the <tbody> with <VarTableRow/> children.

    children -> table contents (header + tbody)
    legend   -> JSX rendered in the legend strip below the table
    summary  -> JSX rendered in the summary box below the legend
  */
  return (
    <div>
      <table class="var_table">
        {children}
      </table>
      {legend && <div class="var_legend">{legend}</div>}
      {summary && <div class="var_summary_box">{summary}</div>}
    </div>
  )
}

export function VarTableHeader({ cols, bulkCells }) {
  /*
    Two-row <thead>: column labels on top, bulk-action cells below.

    cols      -> [{label, className}, ...] one entry per column
    bulkCells -> array of JSX nodes (same length as cols); use null for empty
  */
  return (
    <thead>
      <tr>
        {cols.map((c, i) => (
          <th key={i} scope="col" class={c.className}>{c.label}</th>
        ))}
      </tr>
      <tr>
        {bulkCells.map((cell, i) => (
          <th class="center" key={i} scope="col">{cell}</th>
        ))}
      </tr>
    </thead>
  )
}

export function VarTableRow({ variable, dotBlue, dotGreen, textClass, prefix, cells }) {
  /*
    One body row: name <td> with status dot + emoji prefix, then caller-supplied <td>s.

    variable  -> variable object (uses .name)
    dotBlue   -> bool, applies td_blue to the status dot
    dotGreen  -> bool, applies td_green to the status dot
    textClass -> optional extra class on the name span
    prefix    -> string prepended to variable.name (e.g. emoji markers)
    cells     -> array of <td>…</td> JSX nodes for the rest of the row
  */
  return (
    <tr>
      <td class="row_name">
        <div class="row_name_inner">
          <div class={`table_dot ${dotBlue ? "td_blue" : ""} ${dotGreen ? "td_green" : ""}`} />
          <span class={`table_text ${textClass || ""}`}>{`${prefix || ""}${variable.name}`}</span>
        </div>
      </td>
      {cells}
    </tr>
  )
}
