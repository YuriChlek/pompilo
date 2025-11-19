import { prisma } from "@/lib/prisma";

export async function POST(req: Request) {
    return Response.json({message: 'Hello world'});
}
