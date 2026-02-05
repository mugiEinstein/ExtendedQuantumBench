# LaTeX integration (generic template-safe)

## Recommended include pattern (single-column)
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/Fig2_Typewise_Gating.pdf}
  \caption{...}
  \label{fig:fig2}
\end{figure}
```

## Two-column / wide figure
If you use a two-column template and want full-width:
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/Fig5_OpenEnded_Framework.pdf}
  \caption{...}
\end{figure*}
```

## Notes
- Prefer the **PDF** versions for vector text/lines.
- All figures are saved with `bbox_inches=tight` + small padding, to prevent clipping.
- If your template uses very small caption fonts, consider:
  `\usepackage[font=small,labelfont=bf]{caption}`
