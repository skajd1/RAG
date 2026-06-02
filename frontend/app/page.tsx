"use client";

import { type ChangeEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Copy,
  Database,
  FileText,
  HelpCircle,
  Image as ImageIcon,
  Link as LinkIcon,
  Loader2,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  RefreshCw,
  Search,
  Send,
  Save,
  Square,
  Sun,
  Trash,
  Trash2,
  User,
  X,
  AtSign,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const MarkdownComponents = {
  h1: ({ ...props }: any) => <h1 className="mb-4 mt-6 border-b border-[#E2E3E4] pb-2 text-xl font-black text-[#202124] dark:border-[#303234] dark:text-[#F1F1F1]" {...props} />,
  h2: ({ ...props }: any) => <h2 className="mb-3 mt-5 text-lg font-bold text-[#202124] dark:text-[#F1F1F1]" {...props} />,
  h3: ({ ...props }: any) => <h3 className="mb-2 mt-4 text-base font-bold text-[#374151] dark:text-[#E5E7EB]" {...props} />,
  p: ({ ...props }: any) => <p className="mb-4 leading-7 text-[#4B5563] dark:text-[#D4D7DB]" {...props} />,
  strong: ({ ...props }: any) => <strong className="font-bold text-[#202124] dark:text-[#F1F1F1]" {...props} />,
  ul: ({ ...props }: any) => <ul className="mb-4 ml-4 list-disc space-y-2 text-[#4B5563] dark:text-[#D4D7DB]" {...props} />,
  ol: ({ ...props }: any) => <ol className="mb-4 ml-4 list-decimal space-y-2 text-[#4B5563] dark:text-[#D4D7DB]" {...props} />,
  li: ({ ...props }: any) => <li className="pl-1 marker:text-[#667085] dark:marker:text-[#A1A1AA]" {...props} />,
  blockquote: ({ ...props }: any) => <blockquote className="my-4 border-l-4 border-[#E2E3E4] bg-[#F1F1EF] py-2 pl-4 text-[#4B5563] dark:border-[#303234] dark:bg-[#242424] dark:text-[#D4D7DB]" {...props} />,
  code: ({ inline, ...props }: any) =>
    inline ? (
      <code className="rounded bg-[#E7E9EA] px-1.5 py-0.5 font-mono text-[0.85em] text-[#4B5563] dark:bg-[#35383A] dark:text-[#E5E7EB]" {...props} />
    ) : (
      <code className="my-4 block overflow-x-auto rounded-lg bg-[#111111] p-4 font-mono text-sm text-[#F1F1F1]" {...props} />
    ),
  table: ({ ...props }: any) => (
    <div className="my-6 overflow-x-auto rounded-lg border border-[#E2E3E4] dark:border-[#303234]">
      <table className="w-full border-collapse text-left text-sm" {...props} />
    </div>
  ),
  thead: ({ ...props }: any) => <thead className="border-b border-[#E2E3E4] bg-[#F1F1EF] font-bold text-[#374151] dark:border-[#303234] dark:bg-[#242424] dark:text-[#E5E7EB]" {...props} />,
  th: ({ ...props }: any) => <th className="px-4 py-3" {...props} />,
  td: ({ ...props }: any) => <td className="border-b border-[#E2E3E4] px-4 py-3 text-[#4B5563] dark:border-[#303234] dark:text-[#D4D7DB]" {...props} />,
  a: ({ ...props }: any) => <a className="font-bold text-[#4B5563] underline decoration-[#D4D7DB] underline-offset-4 hover:text-[#202124] dark:text-[#E5E7EB] dark:decoration-[#737373] dark:hover:text-white" target="_blank" rel="noopener noreferrer" {...props} />,
};

interface Source {
  title: string;
  url: string;
  page_id?: string;
  breadcrumb?: string;
  content_type?: string;
  source_type?: string;
  space?: string;
  space_name?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  mentions?: MentionItem[];
  sources?: Source[];
  status?: string;
  elapsedMs?: number;
  isEdited?: boolean;
  finishReason?: "stopped" | "timeout" | "error" | "length";
}

interface ChatHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

interface ChatSourceReference {
  title?: string;
  url?: string;
  page_id?: string;
  space?: string;
  space_name?: string;
  source_type?: string;
  content_type?: string;
}

interface IngestStatus {
  status: "idle" | "processing" | "completed" | string;
  last_space: string | null;
  source_type?: SourceType | null;
  processed_chunks: number;
  error?: string | null;
}

interface IngestHistory {
  space: string;
  space_name?: string | null;
  source_type?: SourceType;
  chunks: number;
  time: string;
  status: string;
  error?: string | null;
}

type SourceType = "confluence" | "jira";

interface IngestSourceCandidate {
  key: string;
  name: string;
  source_type: SourceType;
}

interface ScheduleTarget {
  space: string;
  space_name?: string | null;
  source_type: SourceType;
}

interface ScheduleResult extends IngestHistory {
  schedule_slot?: string | null;
  message?: string;
}

interface ScheduleRunStatus {
  status: string;
  current?: ScheduleTarget | null;
  total: number;
  completed: number;
  failed: number;
  results: ScheduleResult[];
  started_at?: string | null;
  completed_at?: string | null;
  last_slot?: string | null;
  last_checked_at?: string | null;
  last_error?: string | null;
}

interface ScheduleOverview {
  scheduler: ScheduleRunStatus;
  schedule_times: string[];
  schedule_timezone: string;
  schedule_now: string;
  due_slot?: string | null;
  scheduler_enabled: boolean;
  tracked_sources: ScheduleTarget[];
  last_runs: string[];
}

interface MentionItem {
  mention_type: "space";
  source_type: SourceType;
  space?: string;
  space_name?: string;
  title: string;
  subtitle?: string;
  content_type?: string;
  url?: string;
}

const initialMessage: Message = {
  role: "assistant",
  content: "안녕하세요. MetsaBrain입니다. 사내 Confluence 지식을 기반으로 질문에 답변합니다.",
};

const CHAT_STORAGE_KEY = "metsabrain-chat-messages";
const SOURCE_PREVIEW_LIMIT = 4;
const CHAT_HISTORY_TURN_LIMIT = 8;
const CHAT_HISTORY_USER_CHAR_LIMIT = 700;
const CHAT_HISTORY_ASSISTANT_CHAR_LIMIT = 900;
const CHAT_HISTORY_SOURCE_LIMIT = 6;

const sourceTypeLabels: Record<SourceType, string> = {
  confluence: "Confluence",
  jira: "Jira",
};

const contentTypeLabels: Record<string, string> = {
  page: "페이지",
  database: "데이터베이스",
  image: "이미지",
  jira_issue: "Jira 이슈",
  jira_project_status_summary: "Jira 현황",
  jira_mention_scope_summary: "Jira 조건 요약",
};

function formatElapsedSeconds(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return minutes > 0 ? `${minutes}분 ${remainingSeconds}초` : `${remainingSeconds}초`;
}

function compactTextForHistory(content: string, maxChars: number) {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxChars) return normalized;
  return `${normalized.slice(0, maxChars).trim()}...`;
}

function compactAssistantForHistory(content: string) {
  const lines = content
    .split(/\n+/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  const meaningfulLines = lines.filter((line) => !/^(출처|참고|제공된|Sources?|Reference)/i.test(line));
  const selected = meaningfulLines.slice(0, 6).join(" ");
  return compactTextForHistory(selected || content, CHAT_HISTORY_ASSISTANT_CHAR_LIMIT);
}

function buildChatHistoryPayload(baseMessages: Message[]): ChatHistoryTurn[] {
  return baseMessages
    .filter((message) => {
      if (!message.content.trim()) return false;
      if (message.role === "assistant" && message.content === initialMessage.content) return false;
      if (message.role === "assistant" && message.finishReason === "error") return false;
      return message.role === "user" || message.role === "assistant";
    })
    .slice(-CHAT_HISTORY_TURN_LIMIT)
    .map((message) => ({
      role: message.role,
      content:
        message.role === "assistant"
          ? compactAssistantForHistory(message.content)
          : compactTextForHistory(message.content, CHAT_HISTORY_USER_CHAR_LIMIT),
    }))
    .filter((message) => message.content);
}

function buildPriorSourcePayload(baseMessages: Message[]): ChatSourceReference[] {
  const sources = baseMessages
    .filter((message) => message.role === "assistant" && Array.isArray(message.sources))
    .flatMap((message) => message.sources || [])
    .filter((source) => source.url || source.page_id || source.title);

  const seen = new Set<string>();
  const unique: ChatSourceReference[] = [];
  for (const source of [...sources].reverse()) {
    const key = source.page_id || source.url || `${source.source_type}:${source.space}:${source.title}`;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push({
      title: source.title,
      url: source.url,
      page_id: source.page_id,
      space: source.space,
      space_name: source.space_name,
      source_type: source.source_type,
      content_type: source.content_type,
    });
    if (unique.length >= CHAT_HISTORY_SOURCE_LIMIT) break;
  }
  return unique.reverse();
}

function renderUserContent(content: string) {
  return content.split(/(@[^\s@]+)/g).map((part, index) => {
    if (part.startsWith("@") && part.length > 1) {
      return (
        <span key={`${part}-${index}`} className="mx-0.5 inline-flex items-center rounded-md bg-white/20 px-1.5 py-0.5 font-black text-white ring-1 ring-white/25">
          {part}
        </span>
      );
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function renderComposerContent(content: string, mentions: MentionItem[]) {
  const selectedTokens = new Map(mentions.map((mention) => [mentionToken(mention).toLowerCase(), mentionLabel(mention)]));
  return content.split(/(@[^\s@]+)/g).map((part, index) => {
    if (selectedTokens.has(part.toLowerCase())) {
      return (
        <span key={`${part}-${index}`} className="rounded bg-[#E7E9EA] text-[#4B5563] dark:bg-[#373A3D] dark:text-[#E5E7EB]">
          {part}
        </span>
      );
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function mentionTokens(content: string) {
  return Array.from(new Set(content.match(/@[^\s@]+/g) || []));
}

function mentionLabel(mention: MentionItem) {
  return mention.space_name || mention.title || mention.space || "";
}

function mentionToken(mention: MentionItem) {
  return `@${mentionLabel(mention).trim().replace(/\s+/g, "_")}`;
}

interface MentionTokenRange {
  start: number;
  end: number;
  mention: MentionItem;
}

function mentionTokenRanges(content: string, mentions: MentionItem[]): MentionTokenRange[] {
  const selectedTokens = new Map(mentions.map((mention) => [mentionToken(mention).toLowerCase(), mention]));
  return Array.from(content.matchAll(/@[^\s@]+/g))
    .map((match) => {
      const mention = selectedTokens.get(match[0].toLowerCase());
      if (!mention || match.index === undefined) return null;
      return { start: match.index, end: match.index + match[0].length, mention };
    })
    .filter((range): range is MentionTokenRange => range !== null);
}

function getAssistantBadge(reason: Message["finishReason"]) {
  if (reason === "stopped") return "중단됨";
  if (reason === "timeout") return "시간 초과";
  if (reason === "length") return "출력 길이 제한";
  if (reason === "error") return "오류";
  return null;
}

function getHistoryStatusLabel(status: string) {
  if (status === "success") return "완료";
  if (status === "indexed") return "저장됨";
  if (status === "failed") return "실패";
  return status;
}

export default function ChatPage() {
  const API_URL = "/api";
  const [messages, setMessages] = useState<Message[]>([initialMessage]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [isClearConfirmOpen, setIsClearConfirmOpen] = useState(false);
  const [isClearingData, setIsClearingData] = useState(false);
  const [clearDataError, setClearDataError] = useState("");
  const [savedSourceType, setSavedSourceType] = useState<SourceType>("confluence");
  const [spaceKey, setSpaceKey] = useState("");
  const [sourceSearchQuery, setSourceSearchQuery] = useState("");
  const [sourceSuggestions, setSourceSuggestions] = useState<IngestSourceCandidate[]>([]);
  const [selectedIngestSource, setSelectedIngestSource] = useState<IngestSourceCandidate | null>(null);
  const [activeSourceSuggestionIndex, setActiveSourceSuggestionIndex] = useState(0);
  const [isSourceSearching, setIsSourceSearching] = useState(false);
  const [sourceSearchError, setSourceSearchError] = useState("");
  const [ingestStatus, setIngestStatus] = useState<IngestStatus>({ status: "idle", last_space: null, processed_chunks: 0 });
  const [history, setHistory] = useState<IngestHistory[]>([]);
  const [scheduleOverview, setScheduleOverview] = useState<ScheduleOverview | null>(null);
  const [scheduleError, setScheduleError] = useState("");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(null);
  const [editingText, setEditingText] = useState("");
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isThemeReady, setIsThemeReady] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  const [reindexMessage, setReindexMessage] = useState("");
  const [ingestStartedAt, setIngestStartedAt] = useState<number | null>(null);
  const [ingestElapsedSeconds, setIngestElapsedSeconds] = useState(0);
  const [isChatRestored, setIsChatRestored] = useState(false);
  const [selectedMentions, setSelectedMentions] = useState<MentionItem[]>([]);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionStart, setMentionStart] = useState<number | null>(null);
  const [mentionSuggestions, setMentionSuggestions] = useState<MentionItem[]>([]);
  const [activeMentionIndex, setActiveMentionIndex] = useState(0);

  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldFollowScrollRef = useRef(true);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mentionListRef = useRef<HTMLDivElement>(null);
  const sourceListRef = useRef<HTMLDivElement>(null);
  const sourceSearchRequestRef = useRef(0);

  const resizeInputTextarea = (textarea: HTMLTextAreaElement) => {
    textarea.style.height = "auto";
    const nextHeight = Math.min(textarea.scrollHeight, 160);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > 160 ? "auto" : "hidden";
  };

  const isChatNearBottom = () => {
    const container = scrollRef.current;
    if (!container) return true;
    return container.scrollHeight - container.scrollTop - container.clientHeight <= 80;
  };

  const handleChatScroll = () => {
    shouldFollowScrollRef.current = isChatNearBottom();
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/ingest/history`);
      if (res.ok) setHistory(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  const fetchScheduleOverview = async () => {
    try {
      const res = await fetch(`${API_URL}/ingest/schedules`);
      if (!res.ok) throw new Error("자동 갱신 상태를 불러오지 못했습니다.");
      setScheduleOverview(await res.json());
      setScheduleError("");
    } catch (e) {
      console.error(e);
      setScheduleError(e instanceof Error ? e.message : "자동 갱신 상태를 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    fetchHistory();
    fetchScheduleOverview();
    const interval = setInterval(fetchScheduleOverview, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    try {
      const savedMessages = localStorage.getItem(CHAT_STORAGE_KEY);
      if (savedMessages) {
        const parsedMessages = JSON.parse(savedMessages);
        if (Array.isArray(parsedMessages) && parsedMessages.length > 0) {
          setMessages(parsedMessages);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsChatRestored(true);
    }
  }, []);

  useEffect(() => {
    if (!isChatRestored) return;
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages.slice(-50)));
  }, [messages, isChatRestored]);

  useEffect(() => {
    const savedTheme = localStorage.getItem("metsabrain-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setIsDarkMode(savedTheme ? savedTheme === "dark" : prefersDark);
    setIsThemeReady(true);
  }, []);

  useEffect(() => {
    if (!isThemeReady) return;
    document.documentElement.classList.toggle("dark", isDarkMode);
    localStorage.setItem("metsabrain-theme", isDarkMode ? "dark" : "light");
  }, [isDarkMode, isThemeReady]);

  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;
    if (ingestStatus.status === "processing") {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/ingest/status`);
          const data = await res.json();
          setIngestStatus(data);
          if (data.status === "completed" || data.status === "failed" || data.status.startsWith("error")) {
            fetchHistory();
            fetchScheduleOverview();
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [ingestStatus.status]);

  useEffect(() => {
    if (scheduleOverview?.scheduler.status !== "processing") return;

    const interval = setInterval(async () => {
      await fetchScheduleOverview();
      await fetchHistory();
    }, 2000);

    return () => clearInterval(interval);
  }, [scheduleOverview?.scheduler.status]);

  useEffect(() => {
    if (ingestStatus.status !== "processing" || !ingestStartedAt) return;

    const updateElapsed = () => setIngestElapsedSeconds(Math.floor((Date.now() - ingestStartedAt) / 1000));
    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [ingestStatus.status, ingestStartedAt]);

  useEffect(() => {
    if (shouldFollowScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (inputRef.current) resizeInputTextarea(inputRef.current);
  }, [input]);

  useEffect(() => {
    if (!mentionQuery.trim()) {
      setMentionSuggestions([]);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API_URL}/search/mentions?q=${encodeURIComponent(mentionQuery.trim())}&limit=20`);
        if (!res.ok) return;
        const data = await res.json();
        setMentionSuggestions(Array.isArray(data) ? data : []);
        setActiveMentionIndex(0);
      } catch (e) {
        console.error(e);
      }
    }, 180);

    return () => clearTimeout(timer);
  }, [mentionQuery]);

  useEffect(() => {
    const requestId = ++sourceSearchRequestRef.current;
    const query = sourceSearchQuery.trim();
    if (!query || selectedIngestSource) {
      setSourceSuggestions([]);
      setSourceSearchError("");
      setIsSourceSearching(false);
      return;
    }

    const controller = new AbortController();
    setIsSourceSearching(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API_URL}/ingest/sources/search?q=${encodeURIComponent(query)}&limit=8`, { signal: controller.signal });
        if (!res.ok) throw new Error("검색 가능한 문서를 불러오지 못했습니다.");
        const data = await res.json();
        if (sourceSearchRequestRef.current !== requestId) return;
        setSourceSuggestions(Array.isArray(data) ? data : []);
        setActiveSourceSuggestionIndex(0);
        setSourceSearchError("");
      } catch (e) {
        if ((e as Error).name === "AbortError" || sourceSearchRequestRef.current !== requestId) return;
        console.error(e);
        setSourceSuggestions([]);
        setSourceSearchError(e instanceof Error ? e.message : "검색 가능한 문서를 불러오지 못했습니다.");
      } finally {
        if (sourceSearchRequestRef.current === requestId) setIsSourceSearching(false);
      }
    }, 250);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [sourceSearchQuery, selectedIngestSource]);

  useEffect(() => {
    const activeItem = mentionListRef.current?.querySelector(`[data-mention-index="${activeMentionIndex}"]`);
    activeItem?.scrollIntoView({ block: "nearest" });
  }, [activeMentionIndex]);

  useEffect(() => {
    const activeItem = sourceListRef.current?.querySelector(`[data-source-index="${activeSourceSuggestionIndex}"]`);
    activeItem?.scrollIntoView({ block: "nearest" });
  }, [activeSourceSuggestionIndex]);

  const updateLastAssistant = (patch: Partial<Message>) => {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, ...patch };
      }
      return next;
    });
  };

  const mentionKey = (mention: MentionItem) => `${mention.source_type}:${mention.space || mention.title}`;

  const updateMentionSearch = (value: string, caret: number) => {
    const beforeCaret = value.slice(0, caret);
    const match = beforeCaret.match(/(^|\s)@([^\s@]{1,30})$/);
    if (!match) {
      setMentionQuery("");
      setMentionStart(null);
      setMentionSuggestions([]);
      return;
    }

    setMentionStart(caret - match[2].length - 1);
    setMentionQuery(match[2]);
  };

  const handleInputChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.target.value;
    resizeInputTextarea(event.target);
    setInput(value);
    setSelectedMentions((prev) => prev.filter((mention) => value.toLowerCase().includes(mentionToken(mention).toLowerCase())));
    updateMentionSearch(value, event.target.selectionStart);
  };

  const removeMentionRange = (range: MentionTokenRange) => {
    const textarea = inputRef.current;
    const before = input.slice(0, range.start).replace(/\s+$/, "");
    const after = input.slice(range.end).replace(/^\s+/, "");
    const nextInput = `${before}${before && after ? " " : ""}${after}`;
    const nextCaret = before.length + (before && after ? 1 : 0);
    setInput(nextInput);
    setSelectedMentions((prev) => prev.filter((mention) => mentionKey(mention) !== mentionKey(range.mention)));
    updateMentionSearch(nextInput, nextCaret);
    requestAnimationFrame(() => {
      textarea?.focus();
      textarea?.setSelectionRange(nextCaret, nextCaret);
    });
  };

  const snapCaretOutOfMention = (textarea: HTMLTextAreaElement) => {
    if (textarea.selectionStart !== textarea.selectionEnd) return;
    const caret = textarea.selectionStart;
    const range = mentionTokenRanges(input, selectedMentions).find((item) => caret > item.start && caret < item.end);
    if (!range) return;
    const nextCaret = caret - range.start < range.end - caret ? range.start : range.end;
    textarea.setSelectionRange(nextCaret, nextCaret);
  };

  const scheduleSnapCaretOutOfMention = (textarea: HTMLTextAreaElement) => {
    requestAnimationFrame(() => snapCaretOutOfMention(textarea));
  };

  const selectMention = (mention: MentionItem) => {
    const textarea = inputRef.current;
    const caret = textarea?.selectionStart ?? input.length;
    const start = mentionStart ?? caret;
    const token = mentionToken(mention);
    const before = input.slice(0, start);
    const after = input.slice(caret).replace(/^\s+/, "");
    const nextInput = `${before}${token}${after ? ` ${after}` : " "}`;
    setInput(nextInput);
    setSelectedMentions((prev) => {
      const key = mentionKey(mention);
      if (prev.some((item) => mentionKey(item) === key)) return prev;
      return [...prev, mention];
    });
    setMentionQuery("");
    setMentionStart(null);
    setMentionSuggestions([]);
    requestAnimationFrame(() => {
      textarea?.focus();
      const nextCaret = before.length + token.length + 1;
      textarea?.setSelectionRange(nextCaret, nextCaret);
    });
  };

  const sendMessage = async (messageText: string, replaceFromIndex?: number, mentionContext: MentionItem[] = selectedMentions) => {
    if ((!messageText.trim() && mentionContext.length === 0) || isLoading) return;

    const userMessage = messageText.trim();
    const baseMessages = replaceFromIndex === undefined ? messages : messages.slice(0, replaceFromIndex);
    const historyPayload = buildChatHistoryPayload(baseMessages);
    const priorSourcesPayload = buildPriorSourcePayload(baseMessages);
    const startedAt = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;

    if (replaceFromIndex === undefined) {
      setInput("");
      setSelectedMentions([]);
      setMentionSuggestions([]);
      setMentionQuery("");
      setMentionStart(null);
    }
    shouldFollowScrollRef.current = true;
    setIsLoading(true);
    setMessages([
      ...baseMessages,
      { role: "user", content: userMessage, mentions: mentionContext, isEdited: replaceFromIndex !== undefined },
      { role: "assistant", content: "", status: "요청을 전송하는 중입니다" },
    ]);

    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    let assistantContent = "";
    let assistantSources: Source[] = [];
    let buffer = "";

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, mentions: mentionContext, history: historyPayload, prior_sources: priorSourcesPayload }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("서버 응답 오류가 발생했습니다.");

      reader = response.body?.getReader();
      if (!reader) throw new Error("응답 스트림을 읽을 수 없습니다.");

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const trimmedPart = part.trim();
          if (!trimmedPart) continue;

          try {
            const data = JSON.parse(trimmedPart);
            if (data.type === "done") {
              const finishReason = ["timeout", "error", "length"].includes(data.finish_reason) ? data.finish_reason : undefined;
              updateLastAssistant({ status: undefined, elapsedMs: performance.now() - startedAt, finishReason });
              await reader.cancel();
              break;
            }
            if (data.type === "status") {
              updateLastAssistant({ status: data.content });
            } else if (data.type === "sources") {
              assistantSources = data.content || [];
              updateLastAssistant({ sources: assistantSources });
            } else if (data.type === "answer") {
              assistantContent += data.content;
              updateLastAssistant({ content: assistantContent, sources: assistantSources });
            }
          } catch {
            // Ignore malformed stream fragments and wait for the next chunk.
          }
        }
      }
    } catch (error: any) {
      if (error?.name === "AbortError") {
        updateLastAssistant({ status: undefined, content: assistantContent || "응답 생성을 중지했습니다.", elapsedMs: performance.now() - startedAt, finishReason: "stopped" });
      } else {
        const isTimeout = `${error?.message || ""}`.toLowerCase().includes("timeout");
        updateLastAssistant({ status: undefined, content: `오류: ${error.message || "알 수 없는 문제가 발생했습니다."}`, finishReason: isTimeout ? "timeout" : "error" });
      }
    } finally {
      if (reader) reader.releaseLock();
      abortRef.current = null;
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    await sendMessage(input);
  };

  const startEditing = (messageIndex: number) => {
    const current = messages[messageIndex];
    if (current?.role !== "user") return;
    setEditingMessageIndex(messageIndex);
    setEditingText(current.content);
  };

  const cancelEditing = () => {
    setEditingMessageIndex(null);
    setEditingText("");
  };

  const resendEditedMessage = async (messageIndex = editingMessageIndex, messageText = editingText) => {
    if (messageIndex === null) return;

    const current = messages[messageIndex];
    if (current?.role !== "user") return;

    const nextQuestion = messageText.trim();
    if (!nextQuestion || nextQuestion === current.content.trim()) return;

    setEditingMessageIndex(null);
    setEditingText("");
    await sendMessage(nextQuestion, messageIndex);
  };

  const stopResponse = () => {
    abortRef.current?.abort();
  };

  const clearIngestSourceSelection = () => {
    setSpaceKey("");
    setSourceSearchQuery("");
    setSourceSuggestions([]);
    setSelectedIngestSource(null);
    setActiveSourceSuggestionIndex(0);
    setIsSourceSearching(false);
    setSourceSearchError("");
  };

  const selectIngestSource = (candidate: IngestSourceCandidate) => {
    setSpaceKey(candidate.key);
    setSourceSearchQuery("");
    setSelectedIngestSource(candidate);
    setSourceSuggestions([]);
    setSourceSearchError("");
  };

  const handleSourceSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" && sourceSuggestions.length > 0) {
      event.preventDefault();
      setActiveSourceSuggestionIndex((prev) => Math.min(prev + 1, sourceSuggestions.length - 1));
      return;
    }
    if (event.key === "ArrowUp" && sourceSuggestions.length > 0) {
      event.preventDefault();
      setActiveSourceSuggestionIndex((prev) => Math.max(prev - 1, 0));
      return;
    }
    if (event.key === "Escape") {
      setSourceSuggestions([]);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      if (sourceSuggestions.length > 0 && !selectedIngestSource) {
        selectIngestSource(sourceSuggestions[activeSourceSuggestionIndex]);
      } else if (selectedIngestSource) {
        handleIngest();
      }
    }
  };

  const handleIngest = async () => {
    if (!selectedIngestSource || !spaceKey.trim() || ingestStatus.status === "processing") return;
    try {
      setIngestStartedAt(Date.now());
      setIngestElapsedSeconds(0);
      setIngestStatus((prev) => ({ ...prev, status: "processing", last_space: spaceKey.trim(), source_type: selectedIngestSource.source_type, error: null, processed_chunks: 0 }));
      const res = await fetch(`${API_URL}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ space_key: spaceKey.trim(), source_type: selectedIngestSource.source_type }),
      });
      if (!res.ok) throw new Error("수집 요청에 실패했습니다.");
      const data = await res.json();
      if (data.status === "busy") throw new Error("다른 문서 저장 작업이 진행 중입니다.");
      clearIngestSourceSelection();
    } catch (e) {
      console.error(e);
      setIngestStatus((prev) => ({ ...prev, status: "failed", error: e instanceof Error ? e.message : "수집 요청에 실패했습니다." }));
    }
  };

  const reindexCurrentDocuments = async () => {
    if (ingestStatus.status === "processing" || history.length === 0) return;
    try {
      setReindexMessage("");
      setIngestStartedAt(Date.now());
      setIngestElapsedSeconds(0);
      setIngestStatus((prev) => ({ ...prev, status: "processing", last_space: "전체", source_type: null, processed_chunks: 0, error: null }));
      const res = await fetch(`${API_URL}/ingest/schedules/run`, { method: "POST" });
      if (!res.ok) throw new Error("전체 재인덱싱 요청에 실패했습니다.");
      const data = await res.json();
      if (data.status === "busy") throw new Error("다른 문서 저장 작업이 진행 중입니다.");
      setReindexMessage("현재 저장된 문서를 다시 인덱싱하고 있습니다.");
      await fetchScheduleOverview();
    } catch (e: any) {
      console.error(e);
      setIngestStatus((prev) => ({ ...prev, status: "failed", error: e instanceof Error ? e.message : "전체 재인덱싱 요청에 실패했습니다." }));
      setReindexMessage(e?.message || "전체 재인덱싱 요청에 실패했습니다.");
    }
  };

  const requestClearAllData = () => {
    setClearDataError("");
    setIsClearConfirmOpen(true);
  };

  const clearAllData = async () => {
    if (isClearingData) return;
    try {
      setIsClearingData(true);
      setClearDataError("");
      const res = await fetch(`${API_URL}/ingest/clear`, { method: "DELETE" });
      if (!res.ok) throw new Error("저장된 문서 초기화에 실패했습니다.");
      await fetchHistory();
      setIsClearConfirmOpen(false);
    } catch (e) {
      setClearDataError(e instanceof Error ? e.message : "저장된 문서 초기화에 실패했습니다.");
    } finally {
      setIsClearingData(false);
    }
  };

  const clearChat = () => setMessages([initialMessage]);

  const copyMessage = async (content: string, index: number) => {
    await navigator.clipboard.writeText(content);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 1200);
  };

  const isInitialChat = messages.length === 1 && messages[0].role === "assistant" && messages[0].content === initialMessage.content && !messages[0].status;
  const scheduleStatus = scheduleOverview?.scheduler;
  const scheduleInProgress = scheduleStatus?.status === "processing";
  const scheduleProgress = scheduleStatus?.total ? Math.round(((scheduleStatus.completed + scheduleStatus.failed) / scheduleStatus.total) * 100) : 0;
  const scheduledCurrentName = scheduleStatus?.current?.space_name || scheduleStatus?.current?.space;
  const ingestStatusLabel = ingestStatus.status === "processing" ? "수집 중" : ingestStatus.status === "completed" ? "수집 완료" : ingestStatus.status === "failed" ? "수집 실패" : ingestStatus.status;
  const ingestStatusDetail = ingestStatus.status === "processing"
    ? formatElapsedSeconds(ingestElapsedSeconds)
    : ingestStatus.processed_chunks
      ? `${ingestStatus.processed_chunks}개`
      : "";
  const guideScheduleLabel = scheduleOverview?.schedule_times?.length
    ? `${scheduleOverview.schedule_times.join(", ")} (${scheduleOverview.schedule_timezone})`
    : "서버 설정 시간";

  return (
    <div className="flex h-screen overflow-hidden bg-[#F7F7F6] text-[#202124] dark:bg-[#111111] dark:text-[#F1F1F1]">
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex flex-col border-r border-[#E2E3E4] bg-white transition-all duration-300 dark:border-[#303234] dark:bg-[#181818] md:relative md:translate-x-0",
          isSidebarCollapsed ? "w-80 md:w-16" : "w-80",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className={cn("flex items-center justify-between border-b border-[#E2E3E4] py-5 dark:border-[#303234]", isSidebarCollapsed ? "px-3 md:justify-center" : "px-6")}>
          <div className={cn("flex items-center gap-3", isSidebarCollapsed && "md:hidden")}>
            <div className="rounded-lg bg-[#E7E9EA] p-2 text-[#667085] dark:bg-[#35383A] dark:text-[#E5E7EB]">
              <Database size={20} />
            </div>
            <div>
              <div className="font-black text-[#202124] dark:text-[#F1F1F1]">MetsaBrain</div>
              <div className="text-xs font-semibold text-[#667085] dark:text-[#A1A1AA]">문서 관리</div>
            </div>
          </div>
          <div className={cn("flex items-center gap-1", isSidebarCollapsed && "md:justify-center")}>
            <button className={cn("hidden rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D] md:flex", isSidebarCollapsed && "bg-[#E7E9EA] dark:bg-[#35383A] dark:text-[#E5E7EB]")} onClick={() => setIsSidebarCollapsed((prev) => !prev)} aria-label={isSidebarCollapsed ? "사이드바 펼치기" : "사이드바 접기"}>
              {isSidebarCollapsed ? <PanelLeftOpen size={20} /> : <PanelLeftClose size={20} />}
            </button>
            <button className="rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D] md:hidden" onClick={() => setIsSidebarOpen(false)} aria-label="사이드바 닫기">
              <X size={20} />
            </button>
          </div>
        </div>

        <div className={cn("custom-scrollbar flex-1 space-y-8 overflow-y-auto p-6 text-xs", isSidebarCollapsed && "md:hidden")}>
          <section>
            <button
              onClick={() => setIsGuideOpen(true)}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#E2E3E4] bg-[#F1F1EF] py-3 text-xs font-black uppercase tracking-wide text-[#4B5563] hover:bg-[#E7E9EA] dark:border-[#303234] dark:bg-[#242424] dark:text-[#E5E7EB] dark:hover:bg-[#2D2D2D]"
            >
              <HelpCircle size={15} />
              RAG 이용 가이드
            </button>
          </section>

          <section className="space-y-3">
            <div className="font-black tracking-wide text-[#667085] dark:text-[#A1A1AA]">문서 추가</div>
            <div className="relative">
              <div className="flex gap-2 rounded-lg border border-[#E2E3E4] bg-white p-1 dark:border-[#303234] dark:bg-[#242424]">
                <div className="flex min-w-0 flex-1 items-center gap-1 px-2">
                  <Search size={14} className="shrink-0 text-[#98A2B3] dark:text-[#737373]" />
                  {selectedIngestSource ? (
                    <div className="flex min-w-0 flex-1 items-center gap-1.5 rounded-md bg-[#F1F1EF] px-2 py-1.5 font-semibold text-[#374151] dark:bg-[#35383A] dark:text-[#E5E7EB]">
                      <span className="truncate">{selectedIngestSource.name}</span>
                      <span className="shrink-0 rounded bg-white px-1.5 py-0.5 text-[10px] font-black uppercase text-[#667085] dark:bg-[#242424] dark:text-[#D4D7DB]">{sourceTypeLabels[selectedIngestSource.source_type]}</span>
                      <span className="shrink-0 rounded bg-white px-1.5 py-0.5 text-[10px] font-black uppercase text-[#667085] dark:bg-[#242424] dark:text-[#D4D7DB]">{selectedIngestSource.key}</span>
                      <button onClick={clearIngestSourceSelection} className="shrink-0 rounded p-0.5 text-[#98A2B3] hover:bg-white hover:text-[#374151] dark:text-[#A1A1AA] dark:hover:bg-[#242424] dark:hover:text-[#E5E7EB]" aria-label="선택 해제">
                        <X size={12} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <input
                        type="text"
                        value={sourceSearchQuery}
                        onChange={(e) => {
                          setSourceSearchQuery(e.target.value);
                          setSpaceKey("");
                          setSourceSuggestions([]);
                          setSelectedIngestSource(null);
                          setSourceSearchError("");
                          setIsSourceSearching(Boolean(e.target.value.trim()));
                        }}
                        onKeyDown={handleSourceSearchKeyDown}
                        placeholder="Confluence 스페이스 또는 Jira 프로젝트 검색"
                        className="min-w-0 flex-1 bg-transparent px-1 py-2 font-semibold text-[#374151] outline-none placeholder:text-[#98A2B3] dark:text-[#E5E7EB] dark:placeholder:text-[#737373]"
                      />
                      {isSourceSearching && <Loader2 size={14} className="shrink-0 animate-spin text-[#667085] dark:text-[#A1A1AA]" />}
                    </>
                  )}
                </div>
                <button
                  onClick={handleIngest}
                  disabled={!selectedIngestSource || ingestStatus.status === "processing"}
                  className="rounded-md bg-[#3F464D] p-2 text-white transition hover:bg-[#374151] active:scale-95 disabled:cursor-not-allowed disabled:bg-[#D4D7DB] dark:bg-[#D4D7DB] dark:text-[#171717] dark:hover:bg-[#E5E7EB] dark:disabled:bg-[#35383A] dark:disabled:text-[#737373]"
                  aria-label="문서 수집 시작"
                >
                  {ingestStatus.status === "processing" ? <RefreshCw size={18} className="animate-spin" /> : <Save size={18} />}
                </button>
              </div>
              {!selectedIngestSource && sourceSearchQuery.trim() && (sourceSuggestions.length > 0 || (!isSourceSearching && !sourceSearchError)) && (
                <div ref={sourceListRef} className="absolute left-0 right-0 top-[calc(100%+4px)] z-20 max-h-56 overflow-y-auto rounded-lg border border-[#E2E3E4] bg-white p-1 shadow-lg dark:border-[#303234] dark:bg-[#242424]">
                  {sourceSuggestions.length > 0 ? sourceSuggestions.map((candidate, index) => (
                    <button
                      key={`${candidate.source_type}:${candidate.key}`}
                      data-source-index={index}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        selectIngestSource(candidate);
                      }}
                      className={cn(
                        "flex w-full items-center justify-between gap-3 rounded-md px-3 py-2 text-left transition",
                        index === activeSourceSuggestionIndex ? "bg-[#F1F1EF] dark:bg-[#35383A]" : "hover:bg-[#F1F1EF] dark:hover:bg-[#35383A]",
                      )}
                    >
                      <span className="truncate font-semibold text-[#374151] dark:text-[#E5E7EB]">{candidate.name}</span>
                      <span className="flex shrink-0 items-center gap-1 text-[10px] font-black uppercase text-[#667085] dark:text-[#A1A1AA]">
                        <span className="rounded bg-[#E5E7EB] px-1.5 py-0.5 dark:bg-[#373A3D]">{sourceTypeLabels[candidate.source_type]}</span>
                        {candidate.key}
                      </span>
                    </button>
                  )) : (
                    <div className="px-3 py-3 text-center font-semibold text-[#667085] dark:text-[#A1A1AA]">검색 결과가 없습니다.</div>
                  )}
                </div>
              )}
            </div>
            {sourceSearchError && <div className="rounded-lg bg-red-50 px-3 py-2 font-semibold text-red-700 dark:bg-red-950/40 dark:text-red-300">{sourceSearchError}</div>}
            {ingestStatus.status !== "idle" && (
              <div
                className={cn(
                  "rounded-lg px-3 py-2.5 font-semibold",
                  ingestStatus.status === "failed"
                    ? "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300"
                    : "bg-[#F1F1EF] text-[#4B5563] dark:bg-[#242424] dark:text-[#D4D7DB]",
                )}
              >
                <div className="flex min-w-0 items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    {ingestStatus.status === "processing" && <Loader2 size={14} className="shrink-0 animate-spin" />}
                    <span className="shrink-0">{ingestStatusLabel}</span>
                    {ingestStatus.last_space && <span className="truncate text-[11px] text-[#667085] dark:text-[#A1A1AA]">{ingestStatus.last_space}</span>}
                  </div>
                  {ingestStatusDetail && <span className="shrink-0 text-[11px] text-[#667085] dark:text-[#A1A1AA]">{ingestStatusDetail}</span>}
                </div>
                {ingestStatus.error ? <div className="mt-1 break-words text-[11px] leading-5">{ingestStatus.error}</div> : null}
              </div>
            )}
          </section>

          <section className="space-y-3">
            <div className="font-black tracking-wide text-[#667085] dark:text-[#A1A1AA]">문서 갱신</div>
            <div className="space-y-2 rounded-lg border border-[#E2E3E4] bg-white p-3 dark:border-[#303234] dark:bg-[#242424]">
              {scheduleInProgress && (
                <div className="space-y-2 rounded-md bg-[#F1F1EF] px-3 py-2 font-semibold text-[#4B5563] dark:bg-[#35383A] dark:text-[#D4D7DB]">
                  <div className="flex min-w-0 items-center justify-between gap-2">
                    <span className="flex min-w-0 items-center gap-2">
                      <Loader2 size={14} className="shrink-0 animate-spin" />
                      <span className="truncate">{scheduledCurrentName || "저장된 문서"}</span>
                    </span>
                    <span className="shrink-0 text-[11px] text-[#667085] dark:text-[#A1A1AA]">{scheduleStatus?.completed || 0}/{scheduleStatus?.total || 0}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-[#E7E9EA] dark:bg-[#242424]">
                    <div className="h-full rounded-full bg-[#667085] transition-all dark:bg-[#D4D7DB]" style={{ width: `${scheduleProgress}%` }} />
                  </div>
                </div>
              )}
              {scheduleError && <div className="break-words rounded-md bg-red-50 px-2 py-1.5 font-semibold leading-5 text-red-700 dark:bg-red-950/40 dark:text-red-300">{scheduleError}</div>}
              {scheduleStatus?.last_error && <div className="break-words rounded-md bg-red-50 px-2 py-1.5 font-semibold leading-5 text-red-700 dark:bg-red-950/40 dark:text-red-300">{scheduleStatus.last_error}</div>}
              <button
                onClick={reindexCurrentDocuments}
                disabled={ingestStatus.status === "processing" || scheduleInProgress || history.length === 0}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-[#3F464D] px-3 py-2.5 font-black text-white transition hover:bg-[#374151] disabled:cursor-not-allowed disabled:bg-[#D4D7DB] dark:bg-[#D4D7DB] dark:text-[#171717] dark:hover:bg-[#E5E7EB] dark:disabled:bg-[#35383A] dark:disabled:text-[#737373]"
              >
                <RefreshCw size={14} className={scheduleInProgress ? "animate-spin" : ""} />
                저장된 문서 다시 수집
              </button>
            </div>
            {reindexMessage && <div className="rounded-lg bg-[#F1F1EF] px-3 py-2 font-semibold text-[#4B5563] dark:bg-[#242424] dark:text-[#E5E7EB]">{reindexMessage}</div>}
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-black tracking-wide text-[#667085] dark:text-[#A1A1AA]">저장된 문서</div>
                <div className="mt-1 text-[11px] font-semibold text-[#667085] dark:text-[#A1A1AA]">휴지통은 전체 벡터 데이터를 삭제합니다.</div>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={requestClearAllData}
                  className="rounded-md p-1.5 text-[#98A2B3] hover:bg-red-50 hover:text-red-500 dark:text-[#A1A1AA] dark:hover:bg-red-950/40 dark:hover:text-red-300"
                  aria-label="저장된 전체 벡터 데이터 삭제"
                  title="저장된 전체 벡터 데이터 삭제"
                >
                  <Trash size={15} />
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 rounded-lg border border-[#E2E3E4] bg-[#F7F7F6] p-1 dark:border-[#303234] dark:bg-[#242424]">
              {(["confluence", "jira"] as SourceType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => setSavedSourceType(type)}
                  className={cn(
                    "rounded-md px-3 py-2 text-[11px] font-black uppercase tracking-wide transition",
                    savedSourceType === type ? "bg-white text-[#374151] shadow-sm dark:bg-[#35383A] dark:text-[#E5E7EB]" : "text-[#667085] hover:text-[#374151] dark:text-[#A1A1AA] dark:hover:text-[#E5E7EB]",
                  )}
                >
                  {sourceTypeLabels[type]}
                </button>
              ))}
            </div>
            {history.filter((item) => (item.source_type || "confluence") === savedSourceType).length === 0 && <div className="rounded-lg border border-dashed border-[#E2E3E4] py-8 text-center text-[#667085] dark:border-[#303234] dark:text-[#A1A1AA]">저장된 문서가 없습니다.</div>}
            {history.filter((item) => (item.source_type || "confluence") === savedSourceType).slice().reverse().map((item, i) => {
              const displayName = item.space_name || item.space;
              return (
                <div key={`${item.space}-${item.time}-${i}`} className="rounded-lg border border-[#E2E3E4] bg-white p-4 dark:border-[#303234] dark:bg-[#242424]">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-black text-[#202124] dark:text-[#F1F1F1]" title={displayName}>{displayName}</div>
                      {item.space_name && item.space_name !== item.space && <div className="mt-1 text-[11px] font-bold uppercase text-[#667085] dark:text-[#A1A1AA]">{item.space}</div>}
                      <div className="mt-1 text-[11px] font-semibold text-[#667085] dark:text-[#A1A1AA]">{item.time}</div>
                    </div>
                    <div className="flex shrink-0 flex-col items-end gap-1">
                      <span className="rounded bg-[#E5E7EB] px-2 py-1 text-[10px] font-black uppercase text-[#4B5563] dark:bg-[#373A3D] dark:text-[#E5E7EB]">{item.source_type || "confluence"}</span>
                      <span className={cn("rounded px-2 py-1 text-[10px] font-black uppercase", item.status === "success" || item.status === "indexed" ? "bg-[#E5E7EB] text-[#4B5563] dark:bg-[#373A3D] dark:text-[#E5E7EB]" : "bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300")}>
                        {getHistoryStatusLabel(item.status)}
                      </span>
                    </div>
                  </div>
                  {item.error && <div className="mt-3 break-words rounded-md bg-red-50 px-2 py-1 text-[11px] font-semibold leading-5 text-red-700 dark:bg-red-950/40 dark:text-red-300">{item.error}</div>}
                  <div className="mt-3 flex items-center gap-2 text-[11px] font-bold text-[#667085] dark:text-[#A1A1AA]">
                    <CheckCircle2 size={13} />
                    {item.chunks}개 조각 저장됨
                  </div>
                </div>
              );
            })}
          </section>
        </div>

        <div className={cn("border-t border-[#E2E3E4] p-6 dark:border-[#303234]", isSidebarCollapsed && "md:hidden")}>
          <button onClick={clearChat} className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#E2E3E4] bg-white py-3 text-xs font-black tracking-wide text-[#4B5563] hover:bg-[#F1F1EF] dark:border-[#303234] dark:bg-[#242424] dark:text-[#E5E7EB] dark:hover:bg-[#2D2D2D]">
            <Trash2 size={15} />
            대화 초기화
          </button>
        </div>

        <div className={cn("hidden flex-1 flex-col items-center gap-3 border-t border-[#E2E3E4] py-4 dark:border-[#303234]", isSidebarCollapsed && "md:flex")}>
          <button onClick={() => setIsGuideOpen(true)} className="rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] hover:text-[#374151] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D] dark:hover:text-[#E5E7EB]" aria-label="RAG 이용 가이드 열기">
            <HelpCircle size={20} />
          </button>
          <button onClick={clearChat} className="rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] hover:text-[#374151] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D] dark:hover:text-[#E5E7EB]" aria-label="대화 초기화">
            <Trash2 size={18} />
          </button>
        </div>
      </aside>

      <main className="relative flex min-w-0 flex-1 flex-col bg-[#F7F7F6] dark:bg-[#111111]">
        <header className="flex items-center justify-between border-b border-[#E2E3E4] bg-white px-5 py-4 dark:border-[#303234] dark:bg-[#111111] md:px-8">
          <div className="flex items-center gap-4">
            <button className="rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D] md:hidden" onClick={() => setIsSidebarOpen(true)} aria-label="사이드바 열기">
              <Menu size={20} />
            </button>
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#E7E9EA] text-[#667085] dark:bg-[#35383A] dark:text-[#E5E7EB]">
              <Bot size={23} />
            </div>
            <div>
              <h1 className="text-base font-black text-[#202124] dark:text-[#F1F1F1]">MetsaBrain</h1>
              <div className="mt-1 flex items-center gap-2 text-xs font-bold text-[#667085] dark:text-[#A1A1AA]">
                <span className="h-2 w-2 rounded-full bg-[#667085] dark:bg-[#A1A1AA]" />
                시스템 정상
              </div>
            </div>
          </div>
          <button
            onClick={() => setIsDarkMode((prev) => !prev)}
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-[#E2E3E4] text-[#667085] hover:bg-[#F1F1EF] hover:text-[#374151] dark:border-[#303234] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D] dark:hover:text-[#E5E7EB]"
            aria-label={isDarkMode ? "라이트 모드로 전환" : "다크 모드로 전환"}
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </header>

        <div ref={scrollRef} onScroll={handleChatScroll} className="custom-scrollbar flex-1 overflow-y-auto p-4 md:p-8">
          <div className="mx-auto max-w-4xl space-y-8">
            {!isInitialChat && messages.map((msg, idx) => (
              <div key={idx} className={cn("flex gap-4", msg.role === "user" ? "flex-row-reverse" : "flex-row")}>
                <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border", msg.role === "user" ? "border-[#E2E3E4] bg-white text-[#667085] dark:border-[#303234] dark:bg-[#242424] dark:text-[#B0B3B8]" : "border-[#E2E3E4] bg-[#E7E9EA] text-[#667085] dark:border-[#303234] dark:bg-[#242424] dark:text-[#B0B3B8]")}>
                  {msg.role === "user" ? <User size={21} /> : <Bot size={21} />}
                </div>

                <div className={cn("flex max-w-[92%] flex-col gap-3", msg.role === "user" ? "items-end" : "items-start")}>
                  <div className={cn("rounded-lg border px-5 py-4 shadow-sm", msg.role === "user" ? "border-[#3F464D] bg-[#3F464D] text-white dark:border-[#303234] dark:bg-[#35383A]" : "border-[#E2E3E4] bg-white text-[#374151] dark:border-[#303234] dark:bg-[#181818] dark:text-[#F1F1F1]")}>
                    {msg.role === "assistant" ? (
                      <>
                        {msg.status && (
                          <div className="mb-3 flex items-center gap-2 rounded-md bg-[#F1F1EF] px-3 py-2 text-xs font-bold text-[#667085] dark:bg-[#242424] dark:text-[#D4D7DB]">
                            <Loader2 size={14} className="animate-spin text-[#667085] dark:text-[#D4D7DB]" />
                            {msg.status}
                          </div>
                        )}
                        {msg.content ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={MarkdownComponents} className="max-w-none break-words font-medium">
                            {msg.content}
                          </ReactMarkdown>
                        ) : (
                          <div className="flex gap-2 py-2">
                            <div className="h-2 w-2 animate-bounce rounded-full bg-[#667085] dark:bg-[#D4D7DB]" />
                            <div className="h-2 w-2 animate-bounce rounded-full bg-[#667085] dark:bg-[#D4D7DB] [animation-delay:-0.15s]" />
                            <div className="h-2 w-2 animate-bounce rounded-full bg-[#667085] dark:bg-[#D4D7DB] [animation-delay:-0.3s]" />
                          </div>
                        )}
                      </>
                    ) : editingMessageIndex === idx ? (
                      <textarea
                        value={editingText}
                        onChange={(e) => setEditingText(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            resendEditedMessage(idx, e.currentTarget.value);
                          }
                          if (e.key === "Escape") {
                            cancelEditing();
                          }
                        }}
                        rows={Math.min(8, Math.max(2, editingText.split("\n").length))}
                        className="min-h-24 w-[min(72vw,38rem)] resize-y rounded-md border border-[#E2E3E4] bg-white px-3 py-2 text-[15px] font-semibold leading-7 text-[#202124] outline-none ring-2 ring-[#D4D7DB] placeholder:text-[#98A2B3] dark:border-[#303234] dark:bg-[#242424] dark:text-[#F1F1F1] dark:ring-[#35383A]"
                        autoFocus
                      />
                    ) : (
                      <div className="whitespace-pre-wrap text-[15px] font-semibold leading-7">{renderUserContent(msg.content)}</div>
                    )}
                  </div>

                  {msg.role === "assistant" && (msg.content || msg.elapsedMs) && (
                    <div className="flex items-center gap-2 px-1 text-[11px] font-bold text-[#667085] dark:text-[#A1A1AA]">
                      {msg.elapsedMs ? <span>{(msg.elapsedMs / 1000).toFixed(1)}s</span> : null}
                      {getAssistantBadge(msg.finishReason) && <span className="rounded bg-[#F1F1EF] px-2 py-1 text-[#4B5563] dark:bg-[#242424] dark:text-[#D4D7DB]">{getAssistantBadge(msg.finishReason)}</span>}
                      {msg.content ? (
                        <button onClick={() => copyMessage(msg.content, idx)} className="flex items-center gap-1 rounded-md px-2 py-1 hover:bg-[#F1F1EF] hover:text-[#374151] dark:hover:bg-[#2D2D2D] dark:hover:text-[#E5E7EB]">
                          {copiedIndex === idx ? <CheckCircle2 size={13} /> : <Copy size={13} />}
                          {copiedIndex === idx ? "복사됨" : "복사"}
                        </button>
                      ) : null}
                    </div>
                  )}

                  {msg.role === "user" && (
                    <div className="flex items-center gap-2 px-1 text-[11px] font-bold text-[#667085] dark:text-[#A1A1AA]">
                      {msg.isEdited && <span className="text-[#667085] dark:text-[#A1A1AA]">수정됨</span>}
                      {editingMessageIndex === idx ? (
                        <>
                          <button
                            onClick={() => resendEditedMessage(idx)}
                            disabled={isLoading || !editingText.trim() || editingText.trim() === msg.content.trim()}
                            className="flex items-center gap-1 rounded-md px-2 py-1 text-[#4B5563] hover:bg-[#F1F1EF] disabled:cursor-not-allowed disabled:opacity-50 dark:text-[#D4D7DB] dark:hover:bg-[#2D2D2D]"
                          >
                            <Send size={13} />
                            전송
                          </button>
                          <button onClick={cancelEditing} className="rounded-md px-2 py-1 text-[#667085] hover:bg-[#F1F1EF] hover:text-[#374151] dark:text-[#A1A1AA] dark:hover:bg-[#2D2D2D] dark:hover:text-[#E5E7EB]">
                            취소
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={() => startEditing(idx)}
                          disabled={isLoading}
                          aria-label="질문 수정"
                          className="flex items-center gap-1 rounded-md px-2 py-1 text-[#667085] hover:bg-[#F1F1EF] hover:text-[#374151] disabled:cursor-not-allowed disabled:opacity-50 dark:text-[#A1A1AA] dark:hover:bg-[#2D2D2D] dark:hover:text-[#E5E7EB]"
                        >
                          <Pencil size={13} />
                          수정
                        </button>
                      )}
                    </div>
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="w-full space-y-2">
                      <div className="grid w-full gap-2 sm:grid-cols-2">
                        {(expandedSources[idx] ? msg.sources : msg.sources.slice(0, SOURCE_PREVIEW_LIMIT)).map((src, i) => {
                        const isDatabase = src.content_type === "database";
                        const isImage = src.content_type === "image";
                        const isJira = src.source_type === "jira" || src.content_type === "jira_issue";
                        const sourceLabel = isJira ? "Jira" : "Confluence";
                        const contentLabel = contentTypeLabels[src.content_type || "page"] || src.content_type || "페이지";
                        const locationLabel = src.space_name && src.space_name !== src.space ? `${src.space_name} (${src.space})` : src.space_name || src.space;
                        return (
                          <a key={`${src.url}-${i}`} href={src.url} target="_blank" rel="noopener noreferrer" className="group rounded-lg border border-[#E2E3E4] bg-white p-3 shadow-sm transition hover:border-[#D4D7DB] hover:bg-[#F1F1EF] dark:border-[#303234] dark:bg-[#181818] dark:hover:border-[#35383A] dark:hover:bg-[#242424]">
                            <div className="flex items-start gap-3">
                              <div className="rounded-md bg-[#E7E9EA] p-2 text-[#667085] dark:bg-[#35383A] dark:text-[#E5E7EB]">
                                {isImage ? <ImageIcon size={16} /> : isDatabase ? <Database size={16} /> : <FileText size={16} />}
                              </div>
                              <div className="min-w-0 flex-1">
                                <div className="line-clamp-2 text-sm font-black text-[#202124] group-hover:text-[#374151] dark:text-[#F1F1F1] dark:group-hover:text-white">{src.title || "제목 없음"}</div>
                                <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-black">
                                  <span className="rounded bg-[#E5E7EB] px-2 py-1 uppercase text-[#4B5563] dark:bg-[#373A3D] dark:text-[#E5E7EB]">{sourceLabel}</span>
                                  <span className="rounded bg-[#E5E7EB] px-2 py-1 text-[#4B5563] dark:bg-[#373A3D] dark:text-[#E5E7EB]">{contentLabel}</span>
                                  {locationLabel && <span className="rounded bg-[#E5E7EB] px-2 py-1 text-[#4B5563] dark:bg-[#373A3D] dark:text-[#E5E7EB]">{locationLabel}</span>}
                                </div>
                                {src.breadcrumb && <div className="mt-1 line-clamp-2 text-[11px] font-semibold leading-5 text-[#667085] dark:text-[#A1A1AA]">{src.breadcrumb}</div>}
                                <div className="mt-2 flex items-center gap-1 text-[11px] font-black uppercase tracking-wide text-[#667085] dark:text-[#A1A1AA]">
                                  <LinkIcon size={12} />
                                  원문 열기
                                </div>
                              </div>
                            </div>
                          </a>
                        );
                        })}
                      </div>
                      {msg.sources.length > SOURCE_PREVIEW_LIMIT && (
                        <button
                          onClick={() => setExpandedSources((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                          className="inline-flex items-center gap-1 rounded-md border border-[#E2E3E4] bg-white px-3 py-2 text-xs font-black text-[#667085] shadow-sm hover:border-[#D4D7DB] hover:text-[#374151] dark:border-[#303234] dark:bg-[#181818] dark:text-[#D4D7DB] dark:hover:border-[#35383A] dark:hover:text-[#E5E7EB]"
                        >
                          {expandedSources[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          {expandedSources[idx] ? "연관 문서 접기" : `연관 문서 ${msg.sources.length - SOURCE_PREVIEW_LIMIT}개 더보기`}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <footer className={cn("p-4 md:p-6", isInitialChat ? "absolute left-0 right-0 top-1/2 -translate-y-1/2 border-0 bg-transparent" : "border-t border-[#E2E3E4] bg-white dark:border-[#303234] dark:bg-[#111111]")}>
          <div className="mx-auto max-w-4xl space-y-2">
            {isInitialChat && (
              <div className="mb-6 text-center">
                <div className="text-2xl font-semibold text-[#374151] dark:text-[#D4D7DB] md:text-3xl">무엇을 도와드릴까요?</div>
              </div>
            )}
            {!isInitialChat && (
            <div className="flex items-center gap-1.5 px-1 text-xs font-bold text-[#667085] dark:text-[#A1A1AA]">
              <AtSign size={13} />
              <span>검색 범위를 특정 스페이스나 Jira 프로젝트로 좁히려면 @를 입력하세요.</span>
            </div>
            )}
            <div className={cn("relative flex flex-wrap items-end gap-2 border border-[#E2E3E4] bg-white p-2 shadow-lg dark:border-[#303234] dark:bg-[#242424]", isInitialChat ? "rounded-3xl" : "rounded-2xl")}>
              {mentionSuggestions.length > 0 && (
                <div className="absolute bottom-full left-0 mb-2 w-full max-w-xl overflow-hidden rounded-lg border border-[#E2E3E4] bg-white shadow-xl dark:border-[#303234] dark:bg-[#181818]">
                  <div className="border-b border-[#E2E3E4] px-3 py-2 text-xs font-black text-[#667085] dark:border-[#303234] dark:text-[#A1A1AA]">검색 범위 선택</div>
                  <div ref={mentionListRef} className="max-h-72 overflow-y-auto py-1">
                    {mentionSuggestions.map((mention, index) => {
                      const isActive = index === activeMentionIndex;
                      const sourceLabel = mention.source_type === "jira" ? "Jira" : "Confluence";
                      const typeLabel = mention.source_type === "jira" ? "프로젝트" : "스페이스";
                      return (
                        <button
                          key={mentionKey(mention)}
                          data-mention-index={index}
                          onMouseDown={(e) => {
                            e.preventDefault();
                            selectMention(mention);
                          }}
                          className={cn("flex w-full items-start gap-3 px-3 py-2 text-left transition", isActive ? "bg-[#E7E9EA] dark:bg-[#35383A]" : "hover:bg-[#F1F1EF] dark:hover:bg-[#2D2D2D]")}
                        >
                          <div className="mt-0.5 rounded-md bg-[#E7E9EA] p-1.5 text-[#667085] dark:bg-[#35383A] dark:text-[#E5E7EB]">
                            <Database size={14} />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-black text-[#202124] dark:text-[#F1F1F1]">{mention.title}</div>
                            <div className="mt-1 flex flex-wrap gap-1 text-[10px] font-black text-[#667085] dark:text-[#A1A1AA]">
                              <span className="rounded bg-[#E5E7EB] px-1.5 py-0.5 uppercase dark:bg-[#373A3D]">{sourceLabel}</span>
                              <span className="rounded bg-[#E5E7EB] px-1.5 py-0.5 dark:bg-[#373A3D]">{typeLabel}</span>
                            </div>
                            {mention.subtitle && <div className="mt-1 truncate text-xs font-semibold text-[#667085] dark:text-[#A1A1AA]">{mention.subtitle}</div>}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            <div className="relative min-h-11 min-w-[12rem] flex-1">
              {input && (
                <div aria-hidden className="pointer-events-none absolute inset-0 whitespace-pre-wrap break-words px-3 py-3 text-sm font-semibold leading-6 text-[#202124] dark:text-[#F1F1F1]">
                  {renderComposerContent(input, selectedMentions)}
                </div>
              )}
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onClick={(e) => scheduleSnapCaretOutOfMention(e.currentTarget)}
                onKeyUp={(e) => scheduleSnapCaretOutOfMention(e.currentTarget)}
                onSelect={(e) => scheduleSnapCaretOutOfMention(e.currentTarget)}
                onKeyDown={(e) => {
                  const textarea = e.currentTarget;
                  const ranges = mentionTokenRanges(input, selectedMentions);
                  if (textarea.selectionStart === textarea.selectionEnd) {
                    const caret = textarea.selectionStart;
                    if (e.key === "Backspace") {
                      const range = ranges.find((item) => caret > item.start && caret <= item.end);
                      if (range) {
                        e.preventDefault();
                        removeMentionRange(range);
                        return;
                      }
                    }
                    if (e.key === "Delete") {
                      const range = ranges.find((item) => caret >= item.start && caret < item.end);
                      if (range) {
                        e.preventDefault();
                        removeMentionRange(range);
                        return;
                      }
                    }
                    if (e.key === "ArrowLeft") {
                      const range = ranges.find((item) => caret > item.start && caret <= item.end);
                      if (range) {
                        e.preventDefault();
                        textarea.setSelectionRange(range.start, range.start);
                        return;
                      }
                    }
                    if (e.key === "ArrowRight") {
                      const range = ranges.find((item) => caret >= item.start && caret < item.end);
                      if (range) {
                        e.preventDefault();
                        textarea.setSelectionRange(range.end, range.end);
                        return;
                      }
                    }
                  }
                  if (mentionSuggestions.length > 0) {
                    if (e.key === "ArrowDown") {
                      e.preventDefault();
                      setActiveMentionIndex((prev) => (prev + 1) % mentionSuggestions.length);
                      return;
                    }
                    if (e.key === "ArrowUp") {
                      e.preventDefault();
                      setActiveMentionIndex((prev) => (prev - 1 + mentionSuggestions.length) % mentionSuggestions.length);
                      return;
                    }
                    if (e.key === "Enter" || e.key === "Tab") {
                      e.preventDefault();
                      selectMention(mentionSuggestions[activeMentionIndex]);
                      return;
                    }
                    if (e.key === "Escape") {
                      e.preventDefault();
                      setMentionSuggestions([]);
                      setMentionQuery("");
                      setMentionStart(null);
                      return;
                    }
                  }
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={isLoading}
                placeholder={isLoading ? "답변을 생성하는 중입니다..." : "질문을 입력하세요. Shift+Enter로 줄바꿈"}
                rows={1}
                className={cn(
                  "relative z-10 max-h-40 min-h-11 w-full resize-none bg-transparent px-3 py-3 text-sm font-semibold leading-6 outline-none placeholder:text-[#98A2B3] disabled:cursor-not-allowed dark:placeholder:text-[#737373]",
                  input ? "text-transparent caret-[#202124] dark:caret-[#F1F1F1]" : "text-[#202124] dark:text-[#F1F1F1]"
                )}
              />
            </div>
            {isLoading ? (
              <button onClick={stopResponse} className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#3F464D] text-white hover:bg-[#374151] dark:bg-[#D4D7DB] dark:text-[#171717] dark:hover:bg-[#E5E7EB]" aria-label="답변 생성 중단">
                <Square size={17} />
              </button>
            ) : (
              <button onClick={handleSend} disabled={!input.trim() && selectedMentions.length === 0} className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#3F464D] text-white transition hover:bg-[#374151] active:scale-95 disabled:cursor-not-allowed disabled:bg-[#D4D7DB] dark:bg-[#D4D7DB] dark:text-[#171717] dark:hover:bg-[#E5E7EB] dark:disabled:bg-[#35383A] dark:disabled:text-[#737373]" aria-label="Send message">
                <Send size={19} />
              </button>
            )}
            </div>
          </div>
        </footer>
      </main>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 999px;
        }
      `}</style>

      {isGuideOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-[#111111]/40 p-4 dark:bg-[#111111]/70" role="dialog" aria-modal="true" aria-labelledby="rag-guide-title">
          <div className="max-h-[88vh] w-full max-w-2xl overflow-y-auto rounded-lg bg-white shadow-2xl dark:bg-[#181818]">
            <div className="flex items-start justify-between border-b border-[#E2E3E4] px-6 py-5 dark:border-[#303234]">
              <div>
                <h2 id="rag-guide-title" className="text-lg font-black text-[#202124] dark:text-[#F1F1F1]">RAG 이용 가이드</h2>
                <p className="mt-1 text-sm font-semibold text-[#667085] dark:text-[#A1A1AA]">사내 문서를 수집하고, 수집된 지식을 기반으로 답변을 받는 흐름입니다.</p>
              </div>
              <button onClick={() => setIsGuideOpen(false)} className="rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D]" aria-label="가이드 닫기">
                <X size={20} />
              </button>
            </div>

            <div className="space-y-6 px-6 py-5 text-sm leading-6 text-[#4B5563] dark:text-[#D4D7DB]">
              <section>
                <h3 className="mb-2 text-sm font-black uppercase tracking-wide text-[#202124] dark:text-[#F1F1F1]">좌측 사이드바</h3>
                <p>좌측 사이드바는 RAG가 참고할 문서를 관리하는 영역입니다. Confluence 스페이스나 Jira 프로젝트를 저장하고, 저장된 문서 목록을 확인하며, 전체 벡터 데이터를 초기화할 수 있습니다.</p>
              </section>

              <section>
                <h3 className="mb-2 text-sm font-black uppercase tracking-wide text-[#202124] dark:text-[#F1F1F1]">데이터 수집</h3>
                <ol className="list-decimal space-y-2 pl-5">
                  <li>사이드바의 문서 추가 검색창에서 Confluence 스페이스 또는 Jira 프로젝트 이름을 검색합니다.</li>
                  <li>검색 결과에서 대상을 선택하면 검색창 안에 선택 칩이 표시됩니다.</li>
                  <li>저장 아이콘 버튼을 누르면 백엔드가 문서를 가져와 Qdrant에 저장합니다.</li>
                  <li>저장 중에는 간단한 상태와 경과 시간이 표시되고, 완료 후 저장된 문서 목록에 남습니다.</li>
                </ol>
                <p className="mt-3 rounded-lg bg-[#F1F1EF] px-3 py-2 text-[11px] font-semibold leading-5 text-[#667085] dark:bg-[#242424] dark:text-[#A1A1AA]">
                  자동 문서 갱신은 {guideScheduleLabel} 기준으로 서버가 실행합니다. 사이드바의 문서 갱신 버튼은 저장된 문서를 즉시 다시 수집할 때 사용합니다.
                </p>
              </section>

              <section>
                <h3 className="mb-2 text-sm font-black uppercase tracking-wide text-[#202124] dark:text-[#F1F1F1]">질문과 답변</h3>
                <p>채팅창에 질문을 입력하면 백엔드가 Qdrant에서 관련 문서를 찾고, Ollama LLM이 검색된 문맥만 사용해 답변합니다. 답변 아래 출처 카드에서 실제 Confluence/Jira 원문으로 이동할 수 있습니다.</p>
              </section>

              <section>
                <h3 className="mb-2 text-sm font-black uppercase tracking-wide text-[#202124] dark:text-[#F1F1F1]">수정, 중단, 삭제</h3>
                <ul className="list-disc space-y-2 pl-5">
                  <li>답변 생성 중에는 입력창 오른쪽의 정지 버튼으로 생성을 중단할 수 있습니다.</li>
                  <li>내 질문 아래 `수정` 버튼을 누르면 화면 안에서 질문을 고친 뒤 다시 전송할 수 있습니다.</li>
                  <li>`대화 초기화`는 현재 화면의 대화만 초기화합니다.</li>
                  <li>저장된 문서 영역의 휴지통 버튼은 Qdrant 벡터 데이터를 전체 삭제하므로, 재수집이 필요합니다.</li>
                </ul>
              </section>
            </div>
          </div>
        </div>
      )}

      {isClearConfirmOpen && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-[#111111]/40 p-4 dark:bg-[#111111]/70" role="dialog" aria-modal="true" aria-labelledby="clear-data-title">
          <div className="w-full max-w-md rounded-lg bg-white shadow-2xl dark:bg-[#181818]">
            <div className="flex items-start justify-between border-b border-[#E2E3E4] px-6 py-5 dark:border-[#303234]">
              <div>
                <h2 id="clear-data-title" className="text-lg font-black text-[#202124] dark:text-[#F1F1F1]">저장된 문서 전체 삭제</h2>
                <p className="mt-1 text-sm font-semibold text-[#667085] dark:text-[#A1A1AA]">Qdrant에 저장된 벡터 데이터가 모두 삭제됩니다.</p>
              </div>
              <button
                onClick={() => setIsClearConfirmOpen(false)}
                disabled={isClearingData}
                className="rounded-lg p-2 text-[#667085] hover:bg-[#F1F1EF] disabled:cursor-not-allowed disabled:opacity-50 dark:text-[#B0B3B8] dark:hover:bg-[#2D2D2D]"
                aria-label="삭제 확인 닫기"
              >
                <X size={20} />
              </button>
            </div>

            <div className="space-y-4 px-6 py-5 text-sm leading-6 text-[#4B5563] dark:text-[#D4D7DB]">
              <p>삭제 후에는 등록했던 Confluence 스페이스와 Jira 프로젝트를 다시 인덱싱해야 답변에 사용할 수 있습니다.</p>
              {clearDataError && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 font-semibold text-red-700 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-300">
                  {clearDataError}
                </div>
              )}
            </div>

            <div className="flex justify-end gap-2 border-t border-[#E2E3E4] px-6 py-4 dark:border-[#303234]">
              <button
                onClick={() => setIsClearConfirmOpen(false)}
                disabled={isClearingData}
                className="rounded-lg border border-[#E2E3E4] px-4 py-2 text-sm font-bold text-[#4B5563] hover:bg-[#F1F1EF] disabled:cursor-not-allowed disabled:opacity-50 dark:border-[#303234] dark:text-[#D4D7DB] dark:hover:bg-[#2D2D2D]"
              >
                취소
              </button>
              <button
                onClick={clearAllData}
                disabled={isClearingData}
                className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-bold text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-red-500 dark:hover:bg-red-400"
              >
                {isClearingData && <Loader2 size={16} className="animate-spin" />}
                전체 삭제
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
