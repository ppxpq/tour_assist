#!/usr/bin/env node

import { pathToFileURL } from "node:url";
import path from "node:path";

function readArg(name, fallback = "") {
  const index = process.argv.indexOf(name);
  if (index === -1 || index + 1 >= process.argv.length) {
    return fallback;
  }
  return process.argv[index + 1] || fallback;
}

function printJson(payload, exitCode = 0) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
  process.exit(exitCode);
}

function traceId() {
  const chars = "abcdef0123456789";
  let value = "";
  for (let index = 0; index < 16; index += 1) {
    value += chars[Math.floor(Math.random() * chars.length)];
  }
  return value.toUpperCase();
}

function explainXhsError(payload, hasXsecToken) {
  const code = payload?.code;
  const message = payload?.msg || payload?.message || "";
  if (code === 300011 || message.includes("账号存在异常")) {
    return [
      "小红书接口拒绝了当前请求环境（code 300011）。",
      "这通常不是账号真的坏了，而是 b1、cookie_a1、完整 Cookie、web_session 不是同一个浏览器会话，或笔记链接缺少有效 xsec_token。",
      hasXsecToken ? "请优先重新复制同一浏览器里的 b1、a1 和 Network 请求 Cookie。" : "请尝试从浏览器地址栏复制带 xsec_token 的笔记详情页链接，并重新复制同一浏览器里的 b1、a1 和 Network 请求 Cookie。",
    ].join("");
  }
  if (code === 300031 || message.includes("暂时无法浏览")) {
    return [
      "当前笔记暂时无法浏览（code 300031）。",
      hasXsecToken ? "这通常是笔记不可见、被删除、权限受限，或当前 Cookie 无法访问这条笔记。" : "请优先复制浏览器地址栏里带 xsec_token 的完整笔记链接；如果仍失败，可能是笔记不可见、被删除或当前 Cookie 无法访问。",
    ].join("");
  }
  return message || "";
}

async function importFrom(projectPath, relativePath) {
  const filePath = path.resolve(projectPath, relativePath);
  return import(pathToFileURL(filePath).href);
}

async function main() {
  const projectPath = readArg("--project");
  const noteId = readArg("--note-id");
  const xsecToken = readArg("--xsec-token");
  const sourceUrl = readArg("--url");

  if (!projectPath || !noteId) {
    printJson({ success: false, error: "缺少 XHS 项目路径或 note_id。" }, 2);
  }

  const [{ baseURL, b1, my_cookie: cookie, cookie_a1: cookieA1 }, { get_xs_xt }, { x_s_common }] =
    await Promise.all([
      importFrom(projectPath, "config.js"),
      importFrom(projectPath, "sign/X-S/X-S.js"),
      importFrom(projectPath, "sign/X-S-Common/X-S-Common.js"),
    ]);

  if (!cookie || !cookieA1) {
    printJson({ success: false, error: "xhs/config.js 中缺少 cookie_a1 或完整 cookie。" }, 2);
  }

  const apiPath = "/api/sns/web/v1/feed";
  const body = {
    source_note_id: noteId,
    image_formats: ["jpg", "webp", "avif"],
    extra: { need_body_topic: "1" },
    xsec_source: "pc_feed",
    xsec_token: xsecToken || "",
  };

  const sign = await get_xs_xt(apiPath, body, cookieA1);
  const xS = sign["X-s"];
  const xT = sign["X-t"];
  const response = await fetch(`${baseURL || "https://edith.xiaohongshu.com"}${apiPath}`, {
    method: "POST",
    headers: {
      Accept: "application/json, text/plain, */*",
      "Accept-Encoding": "gzip, deflate, br, zstd",
      "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5",
      "Content-Type": "application/json;charset=UTF-8",
      Cookie: cookie,
      Origin: "https://www.xiaohongshu.com",
      Priority: "u=1, i",
      Referer: "https://www.xiaohongshu.com/",
      "Sec-Ch-Ua": "\"Microsoft Edge\";v=\"125\", \"Chromium\";v=\"125\", \"Not.A/Brand\";v=\"24\"",
      "Sec-Ch-Ua-Mobile": "?0",
      "Sec-Ch-Ua-Platform": "Windows",
      "Sec-Fetch-Dest": "empty",
      "Sec-Fetch-Mode": "cors",
      "Sec-Fetch-Site": "same-site",
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
      "X-B3-Traceid": traceId(),
      "X-S": xS,
      "X-T": String(xT),
      "X-S-Common": x_s_common(xS, xT, b1),
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  let payload = null;
  try {
    payload = JSON.parse(text);
  } catch (error) {
    printJson({ success: false, error: `小红书返回非 JSON 内容：${text.slice(0, 160)}` }, 2);
  }

  if (!response.ok || payload?.success === false || payload?.code) {
    const explainedError = explainXhsError(payload, Boolean(xsecToken));
    printJson({
      success: false,
      status: response.status,
      error: explainedError || `小红书接口请求失败：${response.status}`,
      code: payload?.code,
      raw: payload,
    }, 2);
  }

  const item = payload?.data?.items?.[0];
  if (!item?.note_card) {
    printJson({ success: false, error: "未获取到笔记详情。", raw: payload }, 2);
  }

  printJson({
    success: true,
    source: "xhs",
    url: sourceUrl,
    note_id: item.id || noteId,
    note_card: item.note_card,
  });
}

main().catch((error) => {
  printJson({
    success: false,
    error: error?.message || String(error),
    name: error?.name,
    cause: error?.cause
      ? {
          code: error.cause.code,
          errno: error.cause.errno,
          syscall: error.cause.syscall,
          hostname: error.cause.hostname,
        }
      : undefined,
  }, 2);
});
