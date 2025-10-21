import { Locator, Page } from "@playwright/test";

export async function addTestIds(page: Page) {
  await page.evaluate(() => {
    function setIfMatches(el: Element, pattern: string | RegExp, id: string) {
      const text = (el.textContent || "").trim();
      const re = typeof pattern === "string" ? new RegExp(pattern, "i") : pattern;
      if (re.test(text)) el.setAttribute("data-testid", id);
    }

    document.querySelectorAll("button").forEach((btn) => {
      setIfMatches(
        btn,
        /本日のシグナル|当日シグナル|run today signals|統合実行/,
        "run-today-button"
      );
      setIfMatches(btn, /配分|Allocation|配分結果/, "allocation-button");
    });

    document.querySelectorAll("table").forEach((t, i) => {
      if (!t.hasAttribute("data-testid"))
        t.setAttribute("data-testid", `results-table-${i}`);
    });

    // Ensure existing Streamlit dataframes keep a predictable id if present
    document.querySelectorAll('[data-testid="stDataFrame"]').forEach((el, i) => {
      if (!el.hasAttribute("data-testid"))
        el.setAttribute("data-testid", `stDataFrame-${i}`);
    });
  });
}

export async function getRunButton(page: Page): Promise<Locator> {
  // Prefer data-testid, fallback to role-based lookup
  const byTestId = page.getByTestId("run-today-button");
  if ((await byTestId.count()) > 0) return byTestId.first();
  return page
    .getByRole("button", {
      name: /本日のシグナル|当日シグナル|run today signals|統合実行/i,
    })
    .first();
}
