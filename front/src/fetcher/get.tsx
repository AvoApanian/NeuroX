interface GetInter {
  url: string;
  headers?: HeadersInit;
}

export async function GetMethod({ url, headers }: GetInter) {
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error("Loading error");
  return res.json();
}
