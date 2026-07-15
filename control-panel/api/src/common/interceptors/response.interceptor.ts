import { CallHandler, ExecutionContext, Injectable, NestInterceptor } from '@nestjs/common';
import { ResponseData } from '@/common/dto/response.dto';
import { map, Observable } from 'rxjs';
import { HttpArgumentsHost } from '@nestjs/common/interfaces';
import { Response } from 'express';
import { Reflector } from '@nestjs/core';
import { SKIP_RESPONSE_ENVELOPE_KEY } from '@/common/decorators/skip-response-envelope.decorator';

@Injectable()
export class ResponseInterceptor<T> implements NestInterceptor<T, ResponseData<T>> {
    constructor(private readonly reflector: Reflector) {}

    intercept(
        context: ExecutionContext,
        next: CallHandler<T>,
    ): Observable<ResponseData<T>> | Promise<Observable<ResponseData<T>>> {
        const skipEnvelope = this.reflector.getAllAndOverride<boolean>(SKIP_RESPONSE_ENVELOPE_KEY, [
            context.getHandler(),
            context.getClass(),
        ]);

        if (skipEnvelope) {
            return next.handle() as Observable<ResponseData<T>>;
        }

        const ctx: HttpArgumentsHost = context.switchToHttp();
        const response: Response = ctx.getResponse<Response>();

        return next.handle().pipe(map(data => this.formatSuccessResponse(data, response)));
    }

    private formatSuccessResponse(data: T, response: Response): ResponseData<T> {
        const statusCode: number = response.statusCode;

        return {
            success: true,
            statusCode,
            data,
            timestamp: new Date().toISOString(),
        };
    }
}
